using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Object_Pooler : MonoBehaviour
{

    public List<Pool> pools;
    public Dictionary<string, Queue<GameObject>> poolDictionary;

    // Create editor panel for setting up the pooler
    [System.Serializable]
    public class Pool
    {
        public string tag;
        public GameObject prefab;
        public int size;
    }

    // Singleton pattern - limits / unifies class to a single instance
    public static Object_Pooler Instance;
    private void Awake() 
    {
        Instance = this;
    }

    // Start is called before the first frame update
    void Start()
    {

        // Create a new pool dictionary for storing pools
        poolDictionary = new Dictionary<string, Queue<GameObject>>();

        // Iterate through each pool
        foreach (Pool pool in pools)
        {
            
            // Create empty pool (queue of game objects)
            Queue<GameObject> objectPool = new Queue<GameObject>();

            // Iterate up to size of pool, and enqueue these objects into the pool
            for (int i = 0; i < pool.size; i++)
            {
                GameObject obj = Instantiate(pool.prefab);
                obj.SetActive(false);
                objectPool.Enqueue(obj);
            }

            // Add the tag to the dictionary
            poolDictionary.Add(pool.tag, objectPool);
        }
    }

    // Function to spawn a game object from a pool
    public GameObject SpawnFromPool(string tag, Vector3 position, Quaternion rotation)
    {

        // Check that the tag used is valid
        if (!poolDictionary.ContainsKey(tag))
        {
            Debug.LogWarning("Pool with tag " + tag + " doesn't exist");
            return null;
        }

        // Get next pooled item by dequeueing (take from front)
        GameObject objectToSpawn = poolDictionary[tag].Dequeue();

        // Activate and set to target position and rotation
        objectToSpawn.SetActive(true);
        objectToSpawn.transform.position = position;
        objectToSpawn.transform.rotation = rotation;

        // Set to back of queue / pool
        poolDictionary[tag].Enqueue(objectToSpawn);

        // Return the item
        return objectToSpawn;
    }

}
