using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ObjectRandomiser : MonoBehaviour, IRandomiser
{

    // Array to assign GameObjects for generation
    public GameObject[] objectClasses = new GameObject[1];

    [SerializeField] private int noObjects = 3;

    // Start is called before the first frame update
    void Start()
    {
        //Debug.Log(objectClasses.Length);

        for (int i = 0; i < noObjects; i++)
        {
            GameObject go = objectClasses[0];
            go.GetComponent<Collider>().isTrigger = true;
            Instantiate(go); //, pt3D, Quaternion.Euler(-90,0,0)
        }

    }

    // Update is called once per frame
    void Update()
    {
        
    }

    // Implement interface function to randomise
    public void Randomise()
    {
        //RandomTransform();
    }

    // Randomise the transform of the object (position, rotation)
    void RandomTransform()
    {

    }

}
