import qupath.lib.images.servers.LabeledImageServer
import qupath.lib.objects.PathObjects
import qupath.lib.roi.ROIs
import qupath.lib.objects.classes.PathClassFactory
import qupath.lib.roi.RectangleROI
import static qupath.lib.scripting.QP.*
import qupath.lib.images.servers.LabeledImageServer
import qupath.lib.regions.ImagePlane
import qupath.lib.objects.PathCellObject
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.objects.PathObject
import qupath.lib.regions.RegionRequest
import ij.process.ByteProcessor
import qupath.imagej.tools.IJTools
import ij.process.ImageProcessor
import qupath.lib.images.servers.ImageServer
import qupath.lib.objects.PathObject
import qupath.lib.images.PathImage
import qupath.imagej.tools.PathImagePlus
import javax.imageio.ImageIO
import java.awt.Color
import java.awt.image.BufferedImage
import java.awt.*
import java.util.ArrayList
import java.util.List
import java.nio.file.Files
import java.nio.charset.Charset
import qupath.ext.stardist.StarDist2D

def pathModel = 'path to the project directory\he_heavy_augment.pb'

def stardist = StarDist2D.builder(pathModel)
        .threshold(0.3)              // Probability (detection) threshold
        .normalizePercentiles(1, 99) // Percentile normalization
        .pixelSize(0.3)              // Resolution for detection
        .tileSize(1024)              // Specify width & height of the tile used for prediction
        .ignoreCellOverlaps(true)   // Set to true if you don't care if cells expand into one another
        .measureShape()              // Add shape measurements
        .measureIntensity()          // Add cell measurements (in all compartments)
        .includeProbability(true)    // Add probability as a measurement (enables later filtering)
        .simplify(1)                 // Control how polygons are 'simplified' to remove unnecessary vertices
        .doLog()                     // Use this to log a bit more information while running the script
        .build()
        
def outpath = 'D:\\New QuPath project\\'

//for (entry in project.getImageList()){
//print entry.getImageName()

clearAllObjects()
def imageData = getCurrentImageData()
def hierarchy = imageData.getHierarchy()
def server = imageData.getServer()


// Convert to downsample
double downsample = 1.0
def labelServer = new LabeledImageServer.Builder(imageData)
    .useDetections()
    .backgroundLabel(0, ColorTools.WHITE) // Specify background label (usually 0 or 255)
    .addLabel("Cancer",1)
    .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
    //.multichannelOutput(true)
    .build()
    //.useFilter({p -> p.getPathClass() == getPathClass('Cancer')})
    

// Define output path (relative to project)
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def pathOutput_mask = buildFilePath(PROJECT_BASE_DIR, 'results', name, 'export', 'nuclei_mask')
def pathOutput_img = buildFilePath(PROJECT_BASE_DIR, 'results', name, 'export', 'nuclei_img')
mkdirs(pathOutput_mask)
mkdirs(pathOutput_img)

setImageType('BRIGHTFIELD_H_E');
createAnnotationsFromPixelClassifier("tissue_segmentation", 10000000.0, 500000.0)
selectAnnotations()
createAnnotationsFromPixelClassifier("tissue_classification", 100000.0, 500000.0, "SELECT_NEW")
selectObjectsByClassification("Tumor")

def pathObjects = getSelectedObjects()
stardist.detectObjects(imageData, pathObjects)
runObjectClassifier("cell_classification");
runPlugin('qupath.opencv.features.DelaunayClusteringPlugin', '{"distanceThresholdMicrons": 11.0,  "limitByClass": false,  "addClusterMeasurements": true}');
// Get cells
def cells = getCellObjects()

// Extract Delaunay info (should have run clustering plugin already)
def connections = imageData.getProperties().get('OBJECT_CONNECTIONS')

// Assign labels to clusters
int label = 0
for (group in connections.getConnectionGroups()) {
    def allObjects = group.getPathObjects() as Set
    while (!allObjects.isEmpty()) {
        def nextObject = allObjects.iterator().next()
        label++
        def cluster = new HashSet()
        def pending = [nextObject]
        while (!pending.isEmpty()) {
            nextObject = pending.pop()
            addToCluster(group, nextObject, cluster, pending)
        }
        int id = 0
        for (pathObject in cluster) {
            // REMOVE THE OPTIONS YOU DON'T WANT!
            // Show cluster as classification
            //pathObject.setPathClass(getPathClass("Cluster " + label))
            // Show cluster as name
            pathObject.setName(name + "_" + label + "_" + id)
            id++
            // Add cluster as measurement (but don't use this in an object classifier!)
            //pathObject.getMeasurementList().putMeasurement("Cluster", label)
            pathObject.getMeasurementList().close()        
        }
        allObjects.removeAll(cluster) 
    }
}
fireHierarchyUpdate()


saveDetectionMeasurements(PROJECT_BASE_DIR + '\\results\\' + name)


cell_num = 0
for(cell in getDetectionObjects()){
    if ((cell.getPathClass() == getPathClass("Cancer"))){
        cell_num++
    }
}

cell_threshold = 40000/cell_num

for(cell in getDetectionObjects()){
    if ((cell.getPathClass() == getPathClass("Cancer")) && (Math.random() < cell_threshold)){
    
        def region_mask = RegionRequest.createInstance(
            labelServer.getPath(), downsample, cell.getROI())
            
        roi = cell.getROI()
        pathImage = IJTools.convertToImagePlus(server, region_mask)
        int w = (pathImage.getImage().getWidth() / downsample) as int
        int h = (pathImage.getImage().getHeight() / downsample) as int
     
        //def outputPath_mask = buildFilePath(pathOutput_mask)
        def fileMask = new File(pathOutput_mask, cell.getName() + '.txt')
       
        imp = pathImage.getImage()
        
        bpSLICs = Arrays.asList(createObjectMask(pathImage, cell).getPixels())
        FileWriter writer = new FileWriter(fileMask); 
        for(String str: bpSLICs) {
          writer.write(str + System.lineSeparator());
        }
        writer.close()
            
        //def outputPath_mask = buildFilePath(pathOutput_mask, cell.getName() + '.png')
        //writeImageRegion(labelServer, region_mask, outputPath_mask)
        
        def region_img = RegionRequest.createInstance(server.getPath(), downsample, cell.getROI())
        
        def outputPath_img = buildFilePath(pathOutput_img, cell.getName() + '.png')
        
        writeImageRegion(server, region_img, outputPath_img)    
    }
}    

def createObjectMask(PathImage pathImage, PathObject object) {
    //create a byteprocessor that is the same size as the region we are analyzing
    def bp = new ByteProcessor(pathImage.getImage().getWidth(), pathImage.getImage().getHeight())
    //create a value to fill into the "good" area
    bp.setValue(1.0)

    def roi = object.getROI()
    def roiIJ = IJTools.convertToIJRoi(roi, pathImage)
    bp.fill(roiIJ)
    
    return bp
}

void addToCluster(group, pathObject, cluster, pending) {
    if (cluster.add(pathObject)) {
        for (next in group.getConnectedObjects(pathObject)) {
            if (!cluster.contains(next))
                pending.add(next)
        }
    }
}