### This module is for plotting regional maps, containing inset maps showing the geographical context of the region,
### raster data such as DTM, shapefiles, and scatter points inside the map
### Author: Santa Andria (santa.andria@dicea.unipd.it)
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from eomaps import Maps
import rioxarray
import matplotlib.colors as mcolors
from .config import SHP_PATH


class RegionalMap:
    """
    A class to handle regional map visualizations with raster data.
    """

    def __init__(self, region_name, crs, figsize=(5, 6), cb_space=0):
        """cb_space define a blank space for colorbar"""
        self.pad = 0.06  # Pad w.r.t to width
        self.region_name = region_name
        self.cb_space = cb_space
        self.width = figsize[0]
        self.figsize = (self.width, self.width + self.cb_space)
        self.ax = [
            self.pad,
            1 - (self.width - self.pad * self.width) / (self.width + self.cb_space),
            1 - 2 * self.pad,
            (1 - 2 * self.pad) * self.width / (self.width + self.cb_space),
        ]
        self.m = Maps(crs=crs, figsize=figsize, ax=self.ax)

    def setup_base_map(self, region_shp, gridlines=None):
        """
        Set up the base map with region shapefile and optional gridlines.

        Args:
            region_shp (GeoDataFrame): Region shapefile
            gridlines (dict): Dictionary with gridline specifications
                Example: {
                    'longitudes': [11,12,13],
                    'latitudes': [45,46]
                }
        """
        self.m.add_gdf(
            region_shp, facecolor="None", edgecolor="k", linewidth=0.6, zorder=2
        )

        if gridlines:
            self.m.add_gridlines(
                d=[gridlines["longitudes"], gridlines["latitudes"]],
                labels={"fontsize": 14, "offset": 15},
                linewidth=0.5,
                alpha=0.5,
                linestyle="--",
                c="None",
            )

    def add_raster_layer(
        self, raster_data, lon, lat, proj, vmin=None, vmax=None, cmap=None, cb=False
    ):
        """
        Add raster layer with custom styling options.
        """
        m_val = self.m.new_layer("raster")
        m_val.set_data(raster_data.T, x=lon, y=lat, crs=proj)
        m_val.set_shape.raster()

        if cmap is None:
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "shifted", plt.cm.binary(np.linspace(0, 0.8, 256))
            )

        m_val.plot_map(
            vmin=vmin or np.nanmin(raster_data),
            vmax=vmax or np.nanmax(raster_data),
            cmap=cmap,
            zorder=1,
        )
        # TODO: Add colorbar option

    def add_context_inset(
        self, region_shp, country_shp, inset_extent, position=(0.21, 0.83)
    ):
        """
        Add inset map showing region's location in broader context.
        """
        m_i = self.m.new_inset_map(
            xy=(13, 42.5),
            radius=8,
            plot_position=position,
            plot_size=0.2,
            inset_crs=ccrs.Mercator(),
            shape="rectangles",
            boundary=dict(ec="k", lw=1),
        )

        m_i.add_feature.preset.coastline(linewidth=0.3, scale=50)
        m_i.add_gdf(region_shp, facecolor="r", edgecolor="k", linewidth=0.3, zorder=2)
        m_i.add_gdf(
            country_shp.geometry,
            facecolor="grey",
            edgecolor="k",
            zorder=1,
            linewidth=0.3,
        )
        m_i.set_extent(inset_extent, crs=ccrs.PlateCarree())

    def add_scatter(
        self,
        data_df,
        data_col,
        crs=4326,
        temporary=True,
        to_show=["raster", "base"],
        x_col="Lon",
        y_col="Lat",
        s=None,
        cmap=None,
        cb_label=None,
        save_filepath=None,
    ):
        """
        Adds a scatter plot layer to the map.

        Parameters:
        data_df : pd.DataFrame
            A pandas DataFrame containing the data points to plot.
        data_col : str
            The column name in `data_df` whose values will be visualized.
        crs : int, optional, default=4326
            The coordinate reference system (CRS) of the input data.
        temporary : bool, optional, default=True
            If True, the layer will be flushed after displaying to save memory
            and avoid conflicts when recreating the same layer (e.g., in a loop).
        to_show : list of str, optional, default=['raster', 'base']
            The layers to display alongside the scatter plot when `temporary=True`.
        x_col : str, optional, default='Lon'
            The column in `data_df` representing the x-coordinates (longitude).
        y_col : str, optional, default='Lat'
            The column in `data_df` representing the y-coordinates (latitude).
        s : int or array-like, optional, default=None
            The marker size. Can be a single integer or an array of values for each point.
        cmap : str, matplotlib colormap object, or color, optional, default=None
            The colormap for the scatter plot. Can be:
            - A colormap name (e.g., 'viridis', 'plasma')
            - A colormap object (e.g., `matplotlib.cm.viridis`)
            - A single color (e.g., 'red', '#ff5733')
        cb_label : str, optional, default=None
            Label for the colorbar if a colormap is used.

        Notes:
        ------
        - If `temporary=True`, the layer is removed after display to optimize memory.
        - If `temporary=False`, the layer persists.
        """
        if temporary:
            with self.m.new_layer(data_col) as m_val:  # Flush data after
                self._add_scatter(
                    m_val, data_df, data_col, crs, x_col, y_col, s, cmap, cb_label
                )
                self.m.show_layer(*to_show, data_col, clear=True)
                if save_filepath:
                    self.m.savefig(
                        save_filepath,
                        transparent=None,
                        dpi=300,
                        format="png",
                    )
        else:
            m_val = self.m.new_layer(data_col)
            self._add_scatter(
                m_val, data_df, data_col, crs, x_col, y_col, s, cmap, cb_label
            )

    def _add_scatter(
        self, m_val, data_df, data_col, crs, x_col, y_col, s, cmap, cb_label
    ):
        """
        Add inset map showing region's location in broader context.
        """
        vmin, vmax = None, None
        m_val.set_data(
            data=data_df,  # a pandas-DataFrame holding the data & coordinates
            parameter=data_col,
            x=x_col,
            y=y_col,
            crs=crs,  # the coordinate-system of the x- and y- coordinates
        )
        m_val.set_shape.scatter_points(
            size=(
                50 * np.ones_like(data_df[data_col]) if not s else s
            ),  # the marker size in points**2
            marker="o",
        )

        # if cmap is None:
        #     cmap = mcolors.LinearSegmentedColormap.from_list(
        #         "shifted", plt.cm.binary(np.linspace(0, 0.8, 256))
        #     )

        if cmap in mcolors.CSS4_COLORS or mcolors.is_color_like(
            cmap
        ):  # It's a single color
            m_val.plot_map(
                edgecolor="k",
                linewidth=0.3,
                vmin=vmin or np.nanmin(data_df[data_col]),
                vmax=vmax or np.nanmax(data_df[data_col]),
                facecolor=cmap,
                zorder=99,
            )
        else:
            m_val.plot_map(
                edgecolor="k",
                linewidth=0.3,
                vmin=vmin or np.nanmin(data_df[data_col]),
                vmax=vmax or np.nanmax(data_df[data_col]),
                cmap="copper" if not cmap else cmap,
                zorder=99,
            )

        if self.cb_space:
            cb = m_val.add_colorbar(
                pos=(
                    0.05,
                    self.pad * self.width / (self.width + self.cb_space),
                    0.9,
                    (self.cb_space - 2 * self.pad * self.width)
                    / (self.width + self.cb_space),
                ),
                # label=data_col if not cb_label else cb_label,
                # hist_size=1,
                divider_linestyle={"color": "w", "linestyle": "-"},
            )
            cb.tick_params(
                what="histogram",
                left=False,
                bottom=False,
                top=False,
                right=False,
                which="both",
                labelcolor="None",
                grid_color="None",
            )
            cb.tick_params(
                what="colorbar",
                left=False,
                bottom=False,
                top=False,
                right=False,
                which="both",
            )
            cb.set_labels(cb_label=data_col if not cb_label else cb_label, fontsize=14)
            cb.ax_cb.tick_params(labelsize=12)

    def style_map(
        self, extent, compass_pos=(0.9, 0.85), scalebar_pos=(0.1, 0.1), scale=25000
    ):
        """
        Add map styling elements with customizable positions.
        """
        self.m.set_extent(extent, crs=ccrs.PlateCarree())
        self.m.add_compass(pos=compass_pos, style="compass", scale=20)

        self.m.add_scalebar(
            preset="bw",
            n=2,
            auto_position=scalebar_pos,
            rotation=90,
            scale=scale,  # In meters
            scale_props=dict(width=3, colors=("k", "lightgrey")),
            label_props=dict(
                scale=4.5,
                offset=2.5,
                rotation=90,
                every=2,
                family="Fira Sans",
                weight=350,
            ),
            line_props=dict(ec="None", fc="None"),
        )

    def update_cb_space(self, new_cbspace):
        """
        Update cbspace parameter to accomodate colorbar
        """
        self.cb_space = new_cbspace
        ax_pos = [
            self.pad,
            1 - (self.width - self.pad * self.width) / (self.width + self.cb_space),
            1 - 2 * self.pad,
            (1 - 2 * self.pad) * self.width / (self.width + self.cb_space),
        ]
        self.m.ax.set_position(ax_pos)
        self.m.redraw()


def load_raster(path):
    """
    Load rasterr by handling NoData values and getting min/max values.

    Args:
        raster (xarray.DataArray): Tif file

    Returns:
        tuple: (raster_data, lon, lat, crs)
    """
    raster = rioxarray.open_rasterio(path)[0]  # Band 1
    crs = raster.rio.crs.to_epsg()
    lon = raster.x.values
    lat = raster.y.values

    nodata_value = raster.rio.nodata
    if nodata_value is not None:
        raster_data = raster.values
        raster_data = np.ma.masked_equal(raster_data, nodata_value)
    else:
        raster_data = raster.values

    return raster_data, lon, lat, crs


def create_veneto_map(cb=0, show=True, add_inset=True):
    # Space to accomodate colorbar
    VENETO_CONFIG = {
        "extent": [10.3, 13.4, 44.6, 46.7],
        "inset_extent": [6, 19.5, 36, 47.5],  # Italy
        "gridlines": {"longitudes": [11, 12, 13], "latitudes": [45, 46]},
    }

    veneto_shp = gpd.read_file(SHP_PATH / "veneto_6876.shp")
    italy_shp = gpd.read_file(
        SHP_PATH / "georef-italy-regione/georef-italy-regione-millesime.shp"
    )
    dtm_data, lon, lat, proj = load_raster(SHP_PATH / "DTM_6876.tif")

    regional_map = RegionalMap(
        "Veneto", proj, cb_space=cb
    )  # Set cb_space to 0 to not add colorbar
    regional_map.setup_base_map(veneto_shp, VENETO_CONFIG["gridlines"])
    regional_map.add_raster_layer(dtm_data, lon, lat, proj)
    if add_inset:
        regional_map.add_context_inset(
            veneto_shp, italy_shp, VENETO_CONFIG["inset_extent"]
        )

    regional_map.style_map(VENETO_CONFIG["extent"])
    if show:
        regional_map.m.show_layer("raster", "base", clear=True)
    return regional_map