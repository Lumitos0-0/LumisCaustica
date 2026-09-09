package dev.comfyfluffy.caustica.client;

import dev.comfyfluffy.caustica.rt.RtComposite;
import net.minecraft.client.gui.components.debug.DebugScreenDisplayer;
import net.minecraft.client.gui.components.debug.DebugScreenEntries;
import net.minecraft.client.gui.components.debug.DebugScreenEntry;
import net.minecraft.resources.Identifier;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.chunk.LevelChunk;
import org.jspecify.annotations.Nullable;

/**
 * F3 line for the volumetric fog: the live froxel grid, the medium parameters pushed to the shader and
 * which half of the resolved cache the last march read. Off by default like any other optional entry;
 * registration only makes it available in F3's entry list.
 *
 * <p>{@code RtComposite.fogDebugSummaryLine()} owns the displayed values.
 */
public final class RtFogDebugEntry implements DebugScreenEntry {
    public static final Identifier ID = DebugScreenEntries.register(
            Identifier.fromNamespaceAndPath("caustica", "rt_fog"), new RtFogDebugEntry());

    @Override
    public void display(DebugScreenDisplayer displayer, @Nullable Level serverOrClientLevel,
                        @Nullable LevelChunk clientChunk, @Nullable LevelChunk serverChunk) {
        RtComposite composite = RtComposite.INSTANCE;
        if (composite.hasFailed()) {
            return; // vanilla is rendering this frame; the fog state is stale/irrelevant.
        }
        String line = composite.fogDebugSummaryLine();
        if (line != null) {
            displayer.addLine(line);
        }
    }

    @Override
    public boolean isAllowed(boolean reducedDebugInfo) {
        return !reducedDebugInfo;
    }
}
