d3.json("lineage_data.json")
  .then((graphData) => {
    const width = window.innerWidth;
    const height = window.innerHeight;

    const svg = d3
      .select("body")
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .call(
        d3.zoom().on("zoom", (event) => {
          g.attr("transform", event.transform);
        })
      );

    const g = svg.append("g");

    const dropdown = d3.select("#search-bar");
    graphData.nodes.forEach((node) => {
      dropdown.append("option").attr("value", node.id).text(node.id);
    });

    dropdown.on("change", function () {
      const selectedAuthor = this.value;
      if (selectedAuthor) {
        updateGraph(selectedAuthor);
      }
    });

    function updateGraph(selectedAuthor) {
      const ancestors = new Set();
      const descendants = new Set();

      const findAncestors = (id) => {
        graphData.links.forEach((link) => {
          if (link.target === id && !ancestors.has(link.source)) {
            ancestors.add(link.source);
            findAncestors(link.source);
          }
        });
      };

      const findDescendants = (id) => {
        graphData.links.forEach((link) => {
          if (link.source === id && !descendants.has(link.target)) {
            descendants.add(link.target);
            findDescendants(link.target);
          }
        });
      };

      findAncestors(selectedAuthor);
      findDescendants(selectedAuthor);

      const highlightNodes = new Set([
        ...ancestors,
        ...descendants,
        selectedAuthor,
      ]);
      const filteredNodes = graphData.nodes.filter((node) =>
        highlightNodes.has(node.id)
      );
      const filteredLinks = graphData.links.filter(
        (link) =>
          highlightNodes.has(link.source) && highlightNodes.has(link.target)
      );

      // Clear existing graph
      g.selectAll("*").remove();

      // Scale X-axis based on year
      const yearExtent = d3.extent(filteredNodes, (d) => d.year);
      const xScale = d3
        .scaleLinear()
        .domain(yearExtent)
        .range([50, width - 50]);

      // Force simulation layout
      const simulation = d3
        .forceSimulation(filteredNodes)
        .force(
          "link",
          d3
            .forceLink(filteredLinks)
            .id((d) => d.id)
            .distance(100)
        )
        .force("charge", d3.forceManyBody().strength(-300))
        .force(
          "x",
          d3.forceX((d) => xScale(d.year))
        )
        .force("y", d3.forceY(height / 2).strength(0.1))
        .force("center", d3.forceCenter(width / 2, height / 2));

      // Draw links
      const link = g
        .append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(filteredLinks)
        .enter()
        .append("line")
        .attr("class", "link")
        .style("stroke", "gray")
        .style("stroke-width", 1);

      // Draw nodes
      const node = g
        .append("g")
        .attr("class", "nodes")
        .selectAll("circle")
        .data(filteredNodes)
        .enter()
        .append("circle")
        .attr("class", "node")
        .attr("r", 7)
        .style("fill", (d) => {
          if (d.level === "First Gen") return "red";
          if (d.level === "Second Gen") return "blue";
          if (d.level === "Third Gen") return "green";
          if (d.level === "Fourth Gen") return "purple";
          if (d.level === "Fifth Gen") return "orange";
          return "gray";
        });

      const labels = g
        .append("g")
        .selectAll("text")
        .data(filteredNodes)
        .enter()
        .append("text")
        .attr("dy", -10)
        .text((d) => `${d.id} (${d.year})`)
        .attr("text-anchor", "middle");

      simulation.on("tick", () => {
        link
          .attr("x1", (d) => d.source.x)
          .attr("y1", (d) => d.source.y)
          .attr("x2", (d) => d.target.x)
          .attr("y2", (d) => d.target.y);

        node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);

        labels.attr("x", (d) => d.x).attr("y", (d) => d.y);
      });
    }
  })
  .catch((error) => {
    console.error("Error loading the data: ", error);
  });
