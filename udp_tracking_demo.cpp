// Minimal UDP/keyboard/tracking demo for macOS/Linux
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>
#include <cmath>
#include <limits>

// Non-blocking keyboard input
int kbhit() {
    static bool initialized = false;
    static struct termios oldt, newt;
    if (!initialized) {
        tcgetattr(STDIN_FILENO, &oldt);
        newt = oldt;
        newt.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);
        setbuf(stdin, NULL);
        initialized = true;
    }
    int oldf = fcntl(STDIN_FILENO, F_GETFL);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
    int ch = getchar();
    fcntl(STDIN_FILENO, F_SETFL, oldf);
    if (ch != EOF) {
        ungetc(ch, stdin);
        return 1;
    }
    return 0;
}
int getch() { return getchar(); }

// Minimal UDP sender/receiver
class UDPDriver {
    int sockfd;
    std::vector<std::string> clientIPs;
    int socketPort;
    struct sockaddr_in serverAddr;
public:
    UDPDriver() : sockfd(-1), socketPort(0) {}
    void initialise(const char* serverIP, int socketPort) {
        this->socketPort = socketPort;
        sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        memset(&serverAddr, 0, sizeof(serverAddr));
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(socketPort);
        inet_pton(AF_INET, serverIP, &serverAddr.sin_addr);
    }
    unsigned int addClient(const char* clientIP) {
        clientIPs.push_back(std::string(clientIP));
        return clientIPs.size() - 1;
    }
    void sendByte(unsigned char byte, int clientID = -1) {
        struct sockaddr_in destAddr;
        memset(&destAddr, 0, sizeof(destAddr));
        destAddr.sin_family = AF_INET;
        destAddr.sin_port = htons(socketPort);
        if (clientID >= 0 && clientID < (int)clientIPs.size()) {
            inet_pton(AF_INET, clientIPs[clientID].c_str(), &destAddr.sin_addr);
        } else {
            destAddr.sin_addr = serverAddr.sin_addr;
        }
        sendto(sockfd, &byte, 1, 0, (struct sockaddr*)&destAddr, sizeof(destAddr));
    }
    void disconnect() {
        if (sockfd != -1) close(sockfd);
        sockfd = -1;
    }
};

// Tracking data structure
struct TrackerState {
    float position[3];
    float pose[4];
    float headingY;
    struct _OWL_Marker {
        unsigned int ID;
        float x, y, z;
        float timestamp;
    } markers[4];
};

static const int MAX_SIZE_IN_BYTES = sizeof(int) + 12 * (32 * sizeof(char) + sizeof(TrackerState));
char recvBuff[MAX_SIZE_IN_BYTES];

#define PORT 7777
int sockPhaseSpace;
struct sockaddr_in recv_addr;

// Example: 3 robots
const int numMonaRobots = 1;
int Monas[numMonaRobots];

int main() {
    // Setup UDP receive socket
    sockPhaseSpace = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sockPhaseSpace < 0) { perror("socket creation failed"); exit(1); }
    char broadcast = 'a';
    setsockopt(sockPhaseSpace, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast));
    recv_addr.sin_family = AF_INET;
    recv_addr.sin_port = htons(PORT);
    recv_addr.sin_addr.s_addr = INADDR_ANY;
    if (bind(sockPhaseSpace, (struct sockaddr*)&recv_addr, sizeof(recv_addr)) < 0) {
        perror("Error binding UDP socket.");
        close(sockPhaseSpace);
        exit(1);
    }

    // Setup UDP sender
    UDPDriver udp;
    int socketPort = 54007;
    char serverIP[] = "192.168.167.255";
    
    char Mona_addresses[numMonaRobots][14] = {
        "192.168.0.222"
    };
    udp.initialise(serverIP, socketPort);
    for (int i = 0; i < numMonaRobots; i++) {
        Monas[i] = udp.addClient(Mona_addresses[i]);
    }

    printf("Listening for tracking data on UDP port %d. Press keys to send commands. Press x to exit.\n", PORT);

    // Tracking positions for robot and object
    float robot_pos[3] = {std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::quiet_NaN()};
    float object_pos[3] = {std::numeric_limits<float>::quiet_NaN(),
                           std::numeric_limits<float>::quiet_NaN(),
                           std::numeric_limits<float>::quiet_NaN()};
    const char* ROBOT_TRACKER_NAME = "AcoustoBot1";
    const char* OBJECT_TRACKER_NAME = "User1";
    const float REACH_THRESHOLD = 0.05f; // 5 cm

    while (1) {
        // Non-blocking receive
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(sockPhaseSpace, &readfds);
        struct timeval tv = {0, 10000}; // 10ms
        int rv = select(sockPhaseSpace+1, &readfds, NULL, NULL, &tv);
        if (rv > 0 && FD_ISSET(sockPhaseSpace, &readfds)) {
            sockaddr_in sender;
            socklen_t senderLen = sizeof(sender);
            int bytesReceived = recvfrom(sockPhaseSpace, recvBuff, MAX_SIZE_IN_BYTES, 0, (struct sockaddr*)&sender, &senderLen);
            if (bytesReceived > 0) {
                char* parsingIndex = recvBuff;
                int numTrackers = *((int*)parsingIndex);
                parsingIndex += sizeof(int);
                // Reset positions to NaN for this frame
                robot_pos[0] = robot_pos[1] = robot_pos[2] = std::numeric_limits<float>::quiet_NaN();
                object_pos[0] = object_pos[1] = object_pos[2] = std::numeric_limits<float>::quiet_NaN();
                for (int t = 0; t < numTrackers; t++) {
                    char trackerName[33] = {0};
                    memcpy(trackerName, parsingIndex, 32);
                    parsingIndex += 32;
                    TrackerState curState;
                    memcpy(&curState, parsingIndex, sizeof(TrackerState));
                    parsingIndex += sizeof(TrackerState);
                    printf("Tracker: %s | Pos: (%.3f, %.3f, %.3f) | Heading: %.2f\n",
                        trackerName, curState.position[0], curState.position[1], curState.position[2], curState.headingY);
                    if (strcmp(trackerName, ROBOT_TRACKER_NAME) == 0) {
                        memcpy(robot_pos, curState.position, sizeof(robot_pos));
                    }
                    if (strcmp(trackerName, OBJECT_TRACKER_NAME) == 0) {
                        memcpy(object_pos, curState.position, sizeof(object_pos));
                    }
                }
                // After parsing all trackers, check if both positions are valid
                if (!std::isnan(robot_pos[0]) && !std::isnan(object_pos[0])) {
                    float dx = object_pos[0] - robot_pos[0];
                    float dy = object_pos[1] - robot_pos[1];
                    float dz = object_pos[2] - robot_pos[2];
                    float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                    if (dist > REACH_THRESHOLD) {
                        // Move robot forward
                        udp.sendByte('F', Monas[0]);
                        printf("Moving robot towards object. Distance: %.3f\n", dist);
                    } else {
                        // Stop the robot
                        udp.sendByte('S', Monas[0]);
                        printf("Robot reached the object!\n");
                    }
                }
            }
        }
        // Keyboard input
        if (kbhit()) {
            int c = getch();
            if (c == 'x' || c == 'X') break;
            // Example: send command to all robots
            for (int i = 0; i < numMonaRobots; i++) {
                udp.sendByte((unsigned char)c, Monas[i]);
            }
            printf("Sent command '%c' to all robots.\n", c);
        }
    }
    close(sockPhaseSpace);
    udp.disconnect();
    printf("Exited.\n");
    return 0;
} 