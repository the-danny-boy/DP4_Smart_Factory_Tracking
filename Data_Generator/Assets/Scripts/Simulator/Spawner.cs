using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices.ComTypes;
using UnityEngine;
using Random = UnityEngine.Random;

public class Spawner : MonoBehaviour
{

    [SerializeField] GameObject spawnTemplateGameObject;
    [SerializeField] private float COM_height = 0f;
    [SerializeField] private Vector2 spawnExtents = new Vector2(20f, 20f);
    [SerializeField] private int rejectionNumber = 50;
    [SerializeField] private float radius = 10f;
    [SerializeField] private int spawnLimit = 10;

    [SerializeField] private float initial_spawn = 0.0f;
    [SerializeField] private float spawn_period = 1.0f;

    private List<GameObject> spawnedGameObjects = new List<GameObject>();
    private float inclinationAngle = 20f;
    private float displayHeight = 9f;

    public GameObject cameraGameObject;
    Camera cam;

    private float vialCentroidHeight = 0.65f;
    int totalCollisions = 0;

    List<string> trackerData;
    List<int> collisionData;

    private static string pathToWrite = "Assets/Outputs/";
    private static string timestamp = System.DateTime.Now.ToString("yyy.MM.dd_HHmm_");

    private string trackerFile = pathToWrite + timestamp + "trackerData.txt";
    private string collisionFile = pathToWrite + timestamp + "collisionData.txt";
    

    // Start is called before the first frame update
    void Start()
    {

        collisionData = new List<int>();

        cam = cameraGameObject.GetComponent<Camera>();
        Physics.gravity = -9.81f * new Vector3(0f, Mathf.Cos(inclinationAngle * Mathf.Deg2Rad), Mathf.Sin(inclinationAngle * Mathf.Deg2Rad));
        Random.InitState(42);
        InvokeRepeating("Spawn", initial_spawn, spawn_period);

        // TODO - Print out Calibration Info for stitching
        // E.g. the height of camera, object tip height, scale...
        // For scale, can do the inverse transform to find out what 1 pixel corresponds to on the floor

    }

    void OnDrawGizmosSelected()
    {
        Gizmos.color = new Color(0f,1f,1f);
        Gizmos.DrawWireCube(this.transform.position + new Vector3(0f, displayHeight/2f,0f), new Vector3(spawnExtents.x, displayHeight, spawnExtents.y));
    }

    // FixedUpdate is called every physics calculation (0.02s)
    void FixedUpdate()
    {
        // TODO - Print out / write positional information
        // TODO - Correct / map to camera dimensions myself...
        // TODO - Get base measurements, which are at y=0...

        string _positions = "";

         totalCollisions = 0;

        // Iterate through each spawned vial
        foreach (var obj in spawnedGameObjects)
        {

            // Fetch and append object ID
            _positions += obj.GetInstanceID();
            _positions += ",";

            // Finds vial base point by translating down local vial csys by height of centre point
            Vector3 vialBasePt = obj.transform.position + -vialCentroidHeight * obj.transform.forward;
            
            // Append screen position
            Vector3 objScreenPos = cam.WorldToScreenPoint(vialBasePt);
            _positions += new Vector2(objScreenPos.x, objScreenPos.y);
            _positions += " ";


            // Debug.Log(obj.GetComponent<CollisionDetection>().collisionCounter);
            totalCollisions += obj.GetComponent<CollisionDetection>().collisionCounter;

        }

        // Print all the position-id information to the console
        //Debug.Log(_positions);
        //Debug.Log("Total Collisions: " + totalCollisions);

    }

    // Update function - called every time a frame is drawn
    void Update()
    {
        //trackerData.Add();
        //Debug.Log(totalCollisions);
        collisionData.Add(totalCollisions);
        //Debug.Log(collisionData);
    }

    // Function called upon quitting - use this to save out data
    void OnApplicationQuit()
    {

        string writeText = "";
        foreach(int data in collisionData)
        {
            writeText += Convert.ToString(data) + "\n";
        }

        File.AppendAllText(collisionFile, writeText);

    }

    // Spawn function - generates points and instantiates game objects at them
    void Spawn()
    {
        List<Vector2> points = PoissonDiscSample(radius, spawnExtents, rejectionNumber, spawnLimit);
        foreach (Vector2 point in points)
        {
            Vector3 _pt3D = new Vector3(point.x - spawnExtents.x / 2f, COM_height, point.y - spawnExtents.y / 2f) + this.transform.position;
            GameObject _instantiatedObject = Instantiate(spawnTemplateGameObject, _pt3D, Quaternion.Euler(-90,0,0));
            spawnedGameObjects.Add(_instantiatedObject);
        }
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
        float cellSize = _radius / Mathf.Sqrt(2);

        // Initialise background grid of specified extent and resolution
        int[,] backgroundGrid = new int[Mathf.CeilToInt(_spawnExtents.x / cellSize), Mathf.CeilToInt(_spawnExtents.y / cellSize)];

        // Insert a random starting point (x0) into active list
        activeList.Add(_spawnExtents / 2);

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
                float angle = Random.value * Mathf.PI * 2;
                float dist = Mathf.Sqrt(Random.value * (Mathf.Pow(2f * _radius, 2) - Mathf.Pow(_radius, 2)) +
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

}
