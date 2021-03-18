using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices.ComTypes;

using UnityEngine;
using UnityEditor;
using Random = UnityEngine.Random;

public class Spawner_Birdseye : MonoBehaviour
{

    [SerializeField] GameObject spawnTemplateGameObject;
    [SerializeField] private float COM_height = 0f;
    [SerializeField] private Vector2 spawnExtents = new Vector2(20f, 20f);
    [SerializeField] private int rejectionNumber = 50;
    [SerializeField] private float radius = 10f;
    [SerializeField] private int spawnLimit = 10;

    [SerializeField] private int randomSeed = 42;

    [SerializeField] private bool outputWrite = true;
    [SerializeField] private bool tessellate = true;

    [SerializeField] private Vector2 packingStart = new Vector2(0f, 0f);
    [SerializeField] private Vector2 instanceSpacing = new Vector2(0.2f, 0.2f); // 0.2, 0.2
    [SerializeField] private Vector2 instanceShift = new Vector2(-0.02f, 0.1f); // -0.02, 0.1

    [SerializeField] private int colLimit = 40;
    [SerializeField] private int rowLimit = 20;


    [SerializeField] private bool automatedGeneration = true;
    [SerializeField] private int[] counts = {5, 50, 500};
    [SerializeField] private int[] fileNum = {500, 300, 200};
    [SerializeField] private float dataSplit = 0.8f;

    private List<GameObject> spawnedGameObjects = new List<GameObject>();

    public GameObject cameraGameObject;
    Camera cam;

    Object_Pooler objectPooler;

    int runCounter = 0;
    int fileCounter = 0;
    List<float[]> position_data;


    // Start is called before the first frame update
    void Start()
    {
        objectPooler = Object_Pooler.Instance;
        cam = cameraGameObject.GetComponent<Camera>();
        Physics.gravity = new Vector3(0f,0f,0f);
        Random.InitState(randomSeed);
        position_data = new List<float[]>();

        // Create required folders for outputs
        if (!System.IO.Directory.Exists("Data_Generator_Outputs/"))
        {
            Directory.CreateDirectory("Data_Generator_Outputs/");
        }

        foreach(var item in counts)
        {
            if (!System.IO.Directory.Exists("Data_Generator_Outputs/" + item + "/"))
            {
                Directory.CreateDirectory("Data_Generator_Outputs/" + item + "/");

                Directory.CreateDirectory("Data_Generator_Outputs/" + item + "/train/");
                Directory.CreateDirectory("Data_Generator_Outputs/" + item + "/train/images/");
                Directory.CreateDirectory("Data_Generator_Outputs/" + item + "/train/labels/");

                Directory.CreateDirectory("Data_Generator_Outputs/" + item + "/test/");
                Directory.CreateDirectory("Data_Generator_Outputs/" + item + "/test/images/");
                Directory.CreateDirectory("Data_Generator_Outputs/" + item + "/test/labels/");
            }
        }
    }


    // Gizmo draw function to visualise the spawn area
    void OnDrawGizmosSelected()
    {
        Gizmos.color = new Color(0f,1f,1f);
        Gizmos.DrawWireCube(this.transform.position + new Vector3(0f, 1/2f,0f), new Vector3(spawnExtents.x, 1f, spawnExtents.y));
    }


    // Update function - called every time a frame is drawn
    // Used to update the vial positions
    void Update()
    {

        if (automatedGeneration)
        {
            spawnLimit = counts[runCounter];
        }

        List<Vector2> points = new List<Vector2>();

        if(tessellate)
        {
            // Generate new list of points by sampled offset pattern
            points = PatternSample(packingStart, instanceSpacing, instanceShift, spawnExtents, spawnLimit, colLimit, rowLimit);
        }
        else {
            // Generate new list of points using Poisson Disc Sampling
            points = PoissonDiscSample(radius, spawnExtents, rejectionNumber, spawnLimit);
        }

        // Iterate through all points
        for( int i = 0; i < points.Count; i++)
        {
            // Move spawned object to new sampled 3D point
            Vector2 point = points[i];
            Vector3 _pt3D = new Vector3(point.x - spawnExtents.x / 2f, COM_height, point.y - spawnExtents.y / 2f) + this.transform.position;

            // Generate overlap capsule for occupancy test
            Vector3 _pt1 = new Vector3(_pt3D.x, 0.5f, _pt3D.z);
            Vector3 _pt2 = new Vector3(_pt3D.x, 1f, _pt3D.z);
            Collider[] colliders = Physics.OverlapCapsule(_pt1, _pt2, 0.15f);

            // If no collisions, reposition or pull new from pool
            if (colliders.Length == 0)
            {
                if (i < spawnedGameObjects.Count)
                {
                    spawnedGameObjects[i].transform.position = _pt3D;
                    spawnedGameObjects[i].SetActive(true);
                }
                else
                {
                    GameObject _instantiatedObject = objectPooler.SpawnFromPool("Vial", _pt3D, Quaternion.Euler(-90,0,0));
                    spawnedGameObjects.Add(_instantiatedObject);
                }
            }

            // Otherwise deactivate object without repositioning
            else
            {
                if (i < spawnedGameObjects.Count)
                {
                    GameObject item = spawnedGameObjects[i];
                    item.SetActive(false);
                }
            }
        }

        position_data = new List<float[]>();
        for (int i = 0; i < spawnedGameObjects.Count; i++)
        {
            GameObject currentGameObject = spawnedGameObjects[i];
            if(currentGameObject.activeSelf)
            {                
                var renderer = currentGameObject.GetComponent<Renderer>();

                // Get screen point corresponding to vial neck centre
                Vector3 centrePt = renderer.bounds.center + new Vector3(0f, renderer.bounds.extents.y, 0f);
                Vector3 screenCentrePt = cam.WorldToScreenPoint(centrePt);
                Vector2 correctedScreenCentrePt = new Vector2(screenCentrePt.x, cam.pixelHeight-screenCentrePt.y);

                // Get screen point corresponding to corner of bounding box for vial neck (centre + extents)
                Vector3 boundsPt = renderer.bounds.center + new Vector3(renderer.bounds.extents.x, renderer.bounds.extents.y, renderer.bounds.extents.z);
                Vector3 screenBoundsPt = cam.WorldToScreenPoint(boundsPt);
                Vector2 correctedBoundsPt = new Vector2(screenBoundsPt.x, cam.pixelHeight-screenBoundsPt.y);

                // Calculate all YOLO data
                float x = correctedScreenCentrePt.x / cam.pixelWidth;
                float y = correctedScreenCentrePt.y / cam.pixelHeight;
                float width = 2 * (correctedBoundsPt.x - correctedScreenCentrePt.x) / cam.pixelWidth;
                float height = 2 * (correctedScreenCentrePt.y - correctedBoundsPt.y) / cam.pixelHeight;
                float[] yoloData = {0f, x, y, width, height};

                // Check for validity (in range 0-1), and add to list if true
                // Note - can do in the following way as single class id = 0
                if (Mathf.Min(yoloData) >= 0f && Mathf.Max(yoloData) <= 1f)
                    position_data.Add(yoloData);
            }
        }

        if(outputWrite)
        {
            // Format contents of position data to string delimited by newline
            string writeText = "";
            foreach(float[] _data in position_data)
            {
                string dataString = String.Join(" ", _data);
                writeText += dataString + "\n";   
            }

            // Write to file (to train and test with specified split)
            if (fileCounter <= dataSplit * fileNum[runCounter])
                File.WriteAllText("Data_Generator_Outputs/" + counts[runCounter] + "/train/labels/" + fileCounter + ".txt", writeText);
            else
                File.WriteAllText("Data_Generator_Outputs/" + counts[runCounter] + "/test/labels/" + fileCounter + ".txt", writeText);

            // Save out accompanying image
            saveCameraView();
        }

        // Check if written out N=fileNum files
        if (automatedGeneration && fileCounter > fileNum[runCounter])
        {
            // Check if finished scheduled runs
            if(runCounter == counts.Length-1)
            {
                // Conditional compilation to quit application / playmode upon completion
                #if UNITY_EDITOR
                    UnityEditor.EditorApplication.isPlaying = false;
                #else
                    Application.Quit();
                #endif
            }

            fileCounter = 0;
            runCounter++;
        }

    }


    // Save camera view to PNG
    void saveCameraView()
    {
        // Extract render target and modify attributes, e.g. colour space and format
        RenderTexture rt = cam.targetTexture;
        RenderTexture mRt = new RenderTexture(rt.width, rt.height, rt.depth, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
        mRt.antiAliasing = rt.antiAliasing;

        // Assign modified render target
        cam.targetTexture = mRt;
        RenderTexture.active = cam.targetTexture;

        // Manually render camera view to render target
        cam.Render();
 
        // Read the render target
        Texture2D Image = new Texture2D(cam.targetTexture.width, cam.targetTexture.height);
        Image.ReadPixels(new Rect(0, 0, cam.targetTexture.width, cam.targetTexture.height), 0, 0);
        Image.Apply();
 
        // Extract image as bytes (in JPG format), then dispose of object
        var Bytes = Image.EncodeToJPG();
        Destroy(Image);
 
        // Write to file (to train and test with specified split)
        if (fileCounter <= dataSplit * fileNum[runCounter])
            File.WriteAllBytes("Data_Generator_Outputs/" + counts[runCounter] + "/train/images/" + fileCounter + ".jpg", Bytes);
        else
            File.WriteAllBytes("Data_Generator_Outputs/" + counts[runCounter] + "/test/images/" + fileCounter + ".jpg", Bytes);

        fileCounter++;
    }


    // Inspired by the following paper: https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf
    // Function to generate list of sampled points from annular disc
    List<Vector2> PoissonDiscSample(float _radius, Vector2 _spawnExtents, int _rejectionNumber, int _spawnLimit)
    {
        // Initialise some variables
        int spawnCount = 0;
        List<Vector2> points = new List<Vector2>();
        List<Vector2> activeList = new List<Vector2>();

        // Initialise cell size for 2-Dimensional sample space
        float cellSize = _radius / Mathf.Sqrt(2f);

        // Initialise background grid of specified extent and resolution
        int[,] backgroundGrid = new int[Mathf.CeilToInt(_spawnExtents.x / cellSize), Mathf.CeilToInt(_spawnExtents.y / cellSize)];

        // Insert a random starting point (x0) into active list
        activeList.Add(_spawnExtents / 2f);

        // Repeat while valid candidate spaces in active list
        while (activeList.Count > 0 && spawnCount < _spawnLimit)
        {

            // Select random spawn index
            int spawnIndex = Random.Range(0, activeList.Count);
            Vector2 spawnCentre = activeList[spawnIndex];

            // Validity flag for insertion (collision / overlap check)
            bool isValid = false;
            for (int i = 0; i < _rejectionNumber; i++)
            {
                // Generate random point from uniform distribution about annular area
                float angle = Random.value * Mathf.PI * 2f;
                float dist = Mathf.Sqrt(Random.value * (Mathf.Pow(2f * _radius, 2f) - Mathf.Pow(_radius, 2f)) +
                                        Mathf.Pow(_radius, 2));
                Vector2 pos = spawnCentre + new Vector2(Mathf.Cos(angle), Mathf.Sin(angle)) * dist;

                // Check valid point placement (according to spawn radius)
                if (placementCheck(pos, _spawnExtents, cellSize, _radius, points, backgroundGrid))
                {
                    points.Add(pos);
                    activeList.Add(pos);
                    backgroundGrid[(int) (pos.x / cellSize), (int) (pos.y / cellSize)] = points.Count;
                    isValid = true;
                    spawnCount++;
                    break;
                }
            }

            // If not valid, remove from active list; decrements active list count
            if (!isValid)
            {
                activeList.RemoveAt(spawnIndex);
            }
        }

        // Return sampled points
        return points;
    }


    // Function to check validity of point according to spacing criterion
    bool placementCheck(Vector2 position, Vector2 _spawnExtents, float cellSize, float _radius, List<Vector2> points, int[,] backgroundGrid)
    {
        // Check within the background grid area / domain
        if (position.x >= 0 && position.x < _spawnExtents.x && position.y >= 0 && position.y < _spawnExtents.y)
        {
            // Find cell index position
            int cellX = (int)(_spawnExtents.x / cellSize);
            int cellY = (int)(_spawnExtents.y / cellSize);

            // Find surrounding grid locations for search
            int searchStartX = Mathf.Max(0, cellX - 2);
            int searchEndX = Mathf.Min(cellX + 2, backgroundGrid.GetLength(0) - 1);
            int searchStartY = Mathf.Max(0, cellY - 2);
            int searchEndY = Mathf.Min(cellY + 2, backgroundGrid.GetLength(1) - 1);

            // Iterate over search area / domain
            for (int x = searchStartX; x <= searchEndX; x++)
            {
                for (int y = searchStartY; y <= searchEndY; y++)
                {
                    // Check if something at neighbouring cell
                    int pointIndex = backgroundGrid[x, y] - 1;
                    if (pointIndex != -1)
                    {
                        float sqrDistance = (position - points[pointIndex]).sqrMagnitude;
                        if (sqrDistance < Mathf.Pow(_radius, 2))
                        {
                            return false;
                        }
                    }
                }
            }
            return true;
        }
        return false;
    }


    // Function to generate list of points using a packing / tessellation strategy
    List<Vector2> PatternSample(Vector2 packingStart, Vector2 instanceSpacing, Vector2 instanceShift, Vector2 spawnExtents, int spawnLimit, int colLimit, int rowLimit)
    {

        // Initialise some variables
        List<Vector2> points = new List<Vector2>();
        List<Vector2> _points = new List<Vector2>();

        // Create a random offset in range of -0.1 to 0.1 to translate pattern
        Vector2 randomOffset = new Vector2( (float)Random.value / 5f - 0.1f, (float)Random.value / 5f - 0.1f);

        // Find offset position for bottom left of spawner
        Vector2 positionOffset = new Vector2((spawnExtents.x - (instanceSpacing.x + instanceShift.x) * colLimit)/2f,
                                             (spawnExtents.y - (instanceSpacing.y) * rowLimit)/2f);
        
        // Accumulate offsets for starting point
        Vector2 startPt = packingStart + randomOffset + positionOffset;
        
        // Iterate through rows and columns and add point at target location (using parametric packing - hexagonal / square) 
        for (int i = 0; i < colLimit; i++)
        {
            for (int j = 0; j < rowLimit; j++)
            {
                _points.Add(startPt + new Vector2(instanceSpacing.x * i, instanceSpacing.y * j) + 
                                      new Vector2(instanceShift.x * i, instanceShift.y * (i % 2)));
            }
        }

        // Perform reservoir sampling (select N random items from list)
        int selectCount = spawnLimit;
        for (int k = 0; k < _points.Count; k++)
        {
            // Select item with probability = no. of remaining items / total reservoir size
            if(selectCount/(float)_points.Count > Random.value)
            {
                // Add item to the list, and decrement select count
                points.Add(_points[k]);
                selectCount--;
            }
        }

        // Return sampled points
        return points;
        
    }

}
