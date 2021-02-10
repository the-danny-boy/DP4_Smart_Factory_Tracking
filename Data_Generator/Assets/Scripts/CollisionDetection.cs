using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CollisionDetection : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    // Collision Event Handling
    // Note - here it is continuous - contact
    // Could add cooldown, or exclusion (toggle), etc.
    // Also could add id and magnitude info
    void OnCollisionEnter(Collision other)
    {
        if (other.gameObject.name != "Ground")
        {
            Debug.Log("COLLISION!!!");
        }
    }

}
