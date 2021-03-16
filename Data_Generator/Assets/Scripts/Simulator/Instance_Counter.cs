using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Instance_Counter : MonoBehaviour
{

    [SerializeField] private bool write_output = true;

    private List<Renderer> visibleRenderers = new List<Renderer>();
    private Camera cam;
    List<int> instanceData;

    // File output definitions for instanceData file (no of objects per frame)
    private static string pathToWrite = "Assets/Outputs/";
    private static string timestamp = System.DateTime.Now.ToString("yyy.MM.dd_HHmm_");
    private string instanceFile = pathToWrite + timestamp + "instanceData.txt";
    
    // Start is called before the first frame update
    void Start()
    {
        cam = GetComponent<Camera>();
        instanceData = new List<int>();
    }

    // Update is called once per frame
    void Update()
    {
        // Update visibleRenderers list with visible objects
        findVisibleObjects();

        // Add visibleRenderers count (no. of visible instances) to list
        instanceData.Add(visibleRenderers.Count);
    }

    // Function called upon quitting - use this to save out data
    void OnApplicationQuit()
    {
        // If write output enabled
        if (write_output)
        {
            // Save out the instance count data to file
            string writeText = "";
            foreach(int data in instanceData)
                writeText += Convert.ToString(data) + "\n";
            File.AppendAllText(instanceFile, writeText);
        }
    }

    // Function to find visible (mesh renderer) objects in scene
    void findVisibleObjects()
    {
        // Fetch all renderers in scene
        Renderer[] sceneRenderers = FindObjectsOfType<Renderer>();
        
        // Clear visibleRenderers list so only contains current frame data
        visibleRenderers.Clear();

        // Iterate through renderers 
        // Add visible ones from current frame to list (if a vial object)
        for (int i = 0; i < sceneRenderers.Length; i++)
             if (isVisible(sceneRenderers[i]) && sceneRenderers[i].name.Contains("Vial"))
                 visibleRenderers.Add(sceneRenderers[i]);
    }

    // Function to test for visibility of renderer by camera
    bool isVisible(Renderer renderer)
    {
        // Find the frustum plane for camera (i.e. imaging / view plane)
        Plane[] planes = GeometryUtility.CalculateFrustumPlanes(cam);

        // Return boolean flag for whether lies within frustum plane or not (i.e. visible or not)
        return (GeometryUtility.TestPlanesAABB(planes, renderer.bounds)) ? true : false;
    }

}
