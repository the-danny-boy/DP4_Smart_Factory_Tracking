using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CollisionDetection : MonoBehaviour
{

    public int collisionCounter = 0;

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
            //Note - collisionCounter is per vial - only looks locally to itself
            collisionCounter++;

            //Debug.Log(this.GetComponent<Rigidbody>().velocity.magnitude);
            //Debug.Log(other.GetComponent<Rigidbody>().velocity.magnitude);

            // Get relativeVelocity between colliding components (m/s)
            //Debug.Log(other.relativeVelocity.magnitude);

            //Debug.Log("No. of Collisions = " + collisionCounter);
            //Debug.Log("COLLISION!!!");
        }
    }

}
