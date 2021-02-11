using System;
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

    private List<GameObject> spawnedGameObjects = new List<GameObject>();
    private Vector3 gravityDirection = new Vector3(0f, 0f, -9.81f);
    private float displayHeight = 9f;

    public GameObject cameraGameObject;
    Camera cam;

    private float vialCentroidHeight = 0.65f;

    // Start is called before the first frame update
    void Start()
    {

        cam = cameraGameObject.GetComponent<Camera>();
        Physics.gravity = gravityDirection;
        Random.InitState(42);
        Spawn();
        // TODO - Print out Calibration Info for stitching

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

        }

        // Print all the position-id information to the console
        Debug.Log(_positions);

    }

    // Spawn function - generates points and instantiates game objects at them
    void Spawn()
    {
        List<Vector2> points = PoissonDiscSample(radius, spawnExtents, rejectionNumber, spawnLimit);
        foreach (Vector2 point in points)
        {
            Vector3 _pt3D = new Vector3(point.x - spawnExtents.x / 2f, COM_height, point.y - spawnExtents.y / 2f);
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
