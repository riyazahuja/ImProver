/-- A map `f : X → Y` between two topological spaces is said to be **proper** if it is continuous
and, for all `ℱ : Filter X`, any cluster point of `map f ℱ` is the image by `f` of a cluster point
of `ℱ`. -/
@[mk_iff isProperMap_iff_clusterPt, fun_prop]
structure IsProperMap (f : X → Y) extends Continuous f : Prop where
  /-- By definition, if `f` is a proper map and `ℱ` is any filter on `X`, then any cluster point of
  `map f ℱ` is the image by `f` of some cluster point of `ℱ`. -/
  clusterPt_of_mapClusterPt :
    ∀ ⦃ℱ : Filter X⦄, ∀ ⦃y : Y⦄, MapClusterPt y ℱ f → ∃ x, f x = y ∧ ClusterPt x ℱ


/-- By definition, a proper map is continuous. -/
@[fun_prop]
lemma IsProperMap.continuous (h : IsProperMap f) : Continuous f := h.toContinuous


/-- A proper map is closed. -/
lemma IsProperMap.isClosedMap (h : IsProperMap f) : IsClosedMap f := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    h : IsProperMap f
    ⊢ IsClosedMap f
  -/
  rw [isClosedMap_iff_clusterPt]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    h : IsProperMap f
    ⊢ ∀ (s : Set X) (y : Y), MapClusterPt y (Filter.principal s) f → Exists fun x  …
  -/
  exact fun s y ↦ h.clusterPt_of_mapClusterPt (ℱ := 𝓟 s) (y := y)
  /-
    🎉 no goals
  -/


/-- Characterization of proper maps by ultrafilters. -/
lemma isProperMap_iff_ultrafilter : IsProperMap f ↔ Continuous f ∧
    ∀ ⦃𝒰 : Ultrafilter X⦄, ∀ ⦃y : Y⦄, Tendsto f 𝒰 (𝓝 y) → ∃ x, f x = y ∧ 𝒰 ≤ 𝓝 x := by
  -- This is morally trivial since ultrafilters give all the information about cluster points.
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    ⊢ Iff (IsProperMap f) (And (Continuous f) (∀ ⦃𝒰 : Ultrafilter X⦄ ⦃y : Y⦄, Filt …
  -/
  rw [isProperMap_iff_clusterPt]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    ⊢ Iff (And (Continuous f) (∀ ⦃ℱ : Filter X⦄ ⦃y : Y⦄, MapClusterPt y ℱ f → Exis …
  -/
  refine and_congr_right (fun _ ↦ ?_)
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    x✝ : Continuous f
    ⊢ Iff (∀ ⦃ℱ : Filter X⦄ ⦃y : Y⦄, MapClusterPt y ℱ f → Exists fun x => And (Eq  …
  -/
  constructor <;> intro H
    /-
      case mp
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      x✝ : Continuous f
      H : ∀ ⦃ℱ : Filter X⦄ ⦃y : Y⦄, MapClusterPt y ℱ f → Exists fun x => And (Eq (f  …
      ⊢ ∀ ⦃𝒰 : Ultrafilter X⦄ ⦃y : Y⦄, Filter.Tendsto f (↑𝒰) (nhds y) → Exists fun x …
    -/
  · intro 𝒰 y (hY : (Ultrafilter.map f 𝒰 : Filter Y) ≤ _)
    /-
      case mp
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      x✝ : Continuous f
      H : ∀ ⦃ℱ : Filter X⦄ ⦃y : Y⦄, MapClusterPt y ℱ f → Exists fun x => And (Eq (f  …
      𝒰 : Ultrafilter X
      y : Y
      hY : LE.le (↑(Ultrafilter.map f 𝒰)) (nhds y)
      ⊢ Exists fun x => And (Eq (f x) y) (LE.le (↑𝒰) (nhds x))
    -/
    simp_rw [← Ultrafilter.clusterPt_iff] at hY ⊢
    /-
      case mp
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      x✝ : Continuous f
      H : ∀ ⦃ℱ : Filter X⦄ ⦃y : Y⦄, MapClusterPt y ℱ f → Exists fun x => And (Eq (f  …
      𝒰 : Ultrafilter X
      y : Y
      hY : ClusterPt y ↑(Ultrafilter.map f 𝒰)
      ⊢ Exists fun x => And (Eq (f x) y) (ClusterPt x ↑𝒰)
    -/
    exact H hY
    /-
      🎉 no goals
    -/
  · simp_rw [MapClusterPt, ClusterPt, ← Filter.push_pull', map_neBot_iff, ← exists_ultrafilter_iff,
      forall_exists_index]
    /-
      case mpr
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      x✝ : Continuous f
      H : ∀ ⦃𝒰 : Ultrafilter X⦄ ⦃y : Y⦄, Filter.Tendsto f (↑𝒰) (nhds y) → Exists fun …
      ⊢ ∀ ⦃ℱ : Filter X⦄ ⦃y : Y⦄ (x : Ultrafilter X), LE.le (↑x) (Min.min (Filter.co …
    -/
    intro ℱ y 𝒰 hy
    /-
      case mpr
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      x✝ : Continuous f
      H : ∀ ⦃𝒰 : Ultrafilter X⦄ ⦃y : Y⦄, Filter.Tendsto f (↑𝒰) (nhds y) → Exists fun …
      ℱ : Filter X
      y : Y
      𝒰 : Ultrafilter X
      hy : LE.le (↑𝒰) (Min.min (Filter.comap f (nhds y)) ℱ)
      ⊢ Exists fun x => And (Eq (f x) y) (Exists fun u => LE.le (↑u) (Min.min (nhds  …
    -/
    rcases H (tendsto_iff_comap.mpr <| hy.trans inf_le_left) with ⟨x, hxy, hx⟩
    /-
      case mpr.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      x✝ : Continuous f
      H : ∀ ⦃𝒰 : Ultrafilter X⦄ ⦃y : Y⦄, Filter.Tendsto f (↑𝒰) (nhds y) → Exists fun …
      ℱ : Filter X
      y : Y
      𝒰 : Ultrafilter X
      hy : LE.le (↑𝒰) (Min.min (Filter.comap f (nhds y)) ℱ)
      x : X
      hxy : Eq (f x) y
      hx : LE.le (↑𝒰) (nhds x)
      ⊢ Exists fun x => And (Eq (f x) y) (Exists fun u => LE.le (↑u) (Min.min (nhds  …
    -/
    exact ⟨x, hxy, 𝒰, le_inf hx (hy.trans inf_le_right)⟩
    /-
      🎉 no goals
    -/


lemma isProperMap_iff_ultrafilter_of_t2 [T2Space Y] : IsProperMap f ↔ Continuous f ∧
    ∀ ⦃𝒰 : Ultrafilter X⦄, ∀ ⦃y : Y⦄, Tendsto f 𝒰 (𝓝 y) → ∃ x, 𝒰.1 ≤ 𝓝 x :=
  isProperMap_iff_ultrafilter.trans <| and_congr_right fun hc ↦ forall₃_congr fun _𝒰 _y hy ↦
    exists_congr fun x ↦ and_iff_right_of_imp fun h ↦
      tendsto_nhds_unique ((hc.tendsto x).mono_left h) hy


/-- If `f` is proper and converges to `y` along some ultrafilter `𝒰`, then `𝒰` converges to some
`x` such that `f x = y`. -/
lemma IsProperMap.ultrafilter_le_nhds_of_tendsto (h : IsProperMap f) ⦃𝒰 : Ultrafilter X⦄ ⦃y : Y⦄
    (hy : Tendsto f 𝒰 (𝓝 y)) : ∃ x, f x = y ∧ 𝒰 ≤ 𝓝 x :=
  (isProperMap_iff_ultrafilter.mp h).2 hy


/-- The composition of two proper maps is proper. -/
lemma IsProperMap.comp (hf : IsProperMap f) (hg : IsProperMap g) :
    IsProperMap (g ∘ f) := by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : X → Y
    g : Y → Z
    hf : IsProperMap f
    hg : IsProperMap g
    ⊢ IsProperMap (Function.comp g f)
  -/
  refine ⟨by fun_prop, fun ℱ z h ↦ ?_⟩
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : X → Y
    g : Y → Z
    hf : IsProperMap f
    hg : IsProperMap g
    ℱ : Filter X
    z : Z
    h : MapClusterPt z ℱ (Function.comp g f)
    ⊢ Exists fun x => And (Eq (Function.comp g f x) z) (ClusterPt x ℱ)
  -/
  rw [mapClusterPt_comp] at h
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : X → Y
    g : Y → Z
    hf : IsProperMap f
    hg : IsProperMap g
    ℱ : Filter X
    z : Z
    h : MapClusterPt z (Filter.map f ℱ) g
    ⊢ Exists fun x => And (Eq (Function.comp g f x) z) (ClusterPt x ℱ)
  -/
  rcases hg.clusterPt_of_mapClusterPt h with ⟨y, rfl, hy⟩
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : X → Y
    g : Y → Z
    hf : IsProperMap f
    hg : IsProperMap g
    ℱ : Filter X
    y : Y
    hy : ClusterPt y (Filter.map f ℱ)
    h : MapClusterPt (g y) (Filter.map f ℱ) g
    ⊢ Exists fun x => And (Eq (Function.comp g f x) (g y)) (ClusterPt x ℱ)
  -/
  rcases hf.clusterPt_of_mapClusterPt hy with ⟨x, rfl, hx⟩
  /-
    case intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : X → Y
    g : Y → Z
    hf : IsProperMap f
    hg : IsProperMap g
    ℱ : Filter X
    x : X
    hx : ClusterPt x ℱ
    hy : ClusterPt (f x) (Filter.map f ℱ)
    h : MapClusterPt (g (f x)) (Filter.map f ℱ) g
    ⊢ Exists fun x_1 => And (Eq (Function.comp g f x_1) (g (f x))) (ClusterPt x_1 ℱ)
  -/
  use x, rfl
  /-
    🎉 no goals
  -/



/-- If the composition of two continuous functions `g ∘ f` is proper and `f` is surjective,
then `g` is proper. -/
lemma isProperMap_of_comp_of_surj (hf : Continuous f)
    (hg : Continuous g) (hgf : IsProperMap (g ∘ f)) (f_surj : f.Surjective) : IsProperMap g := by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : X → Y
    g : Y → Z
    hf : Continuous f
    hg : Continuous g
    hgf : IsProperMap (Function.comp g f)
    f_surj : Function.Surjective f
    ⊢ IsProperMap g
  -/
  refine ⟨hg, fun ℱ z h ↦ ?_⟩
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : X → Y
    g : Y → Z
    hf : Continuous f
    hg : Continuous g
    hgf : IsProperMap (Function.comp g f)
    f_surj : Function.Surjective f
    ℱ : Filter Y
    z : Z
    h : MapClusterPt z ℱ g
    ⊢ Exists fun x => And (Eq (g x) z) (ClusterPt x ℱ)
  -/
  rw [← ℱ.map_comap_of_surjective f_surj, ← mapClusterPt_comp] at h
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : X → Y
    g : Y → Z
    hf : Continuous f
    hg : Continuous g
    hgf : IsProperMap (Function.comp g f)
    f_surj : Function.Surjective f
    ℱ : Filter Y
    z : Z
    h : MapClusterPt z (Filter.comap f ℱ) (Function.comp g f)
    ⊢ Exists fun x => And (Eq (g x) z) (ClusterPt x ℱ)
  -/
  rcases hgf.clusterPt_of_mapClusterPt h with ⟨x, rfl, hx⟩
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : X → Y
    g : Y → Z
    hf : Continuous f
    hg : Continuous g
    hgf : IsProperMap (Function.comp g f)
    f_surj : Function.Surjective f
    ℱ : Filter Y
    x : X
    hx : ClusterPt x (Filter.comap f ℱ)
    h : MapClusterPt (Function.comp g f x) (Filter.comap f ℱ) (Function.comp g f)
    ⊢ Exists fun x_1 => And (Eq (g x_1) (Function.comp g f x)) (ClusterPt x_1 ℱ)
  -/
  rw [← ℱ.map_comap_of_surjective f_surj]
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : X → Y
    g : Y → Z
    hf : Continuous f
    hg : Continuous g
    hgf : IsProperMap (Function.comp g f)
    f_surj : Function.Surjective f
    ℱ : Filter Y
    x : X
    hx : ClusterPt x (Filter.comap f ℱ)
    h : MapClusterPt (Function.comp g f x) (Filter.comap f ℱ) (Function.comp g f)
    ⊢ Exists fun x_1 => And (Eq (g x_1) (Function.comp g f x)) (ClusterPt x_1 (Fil …
  -/
  exact ⟨f x, rfl, hx.map hf.continuousAt tendsto_map⟩
  /-
    🎉 no goals
  -/


/-- If the composition of two continuous functions `g ∘ f` is proper and `g` is injective,
then `f` is proper. -/
lemma isProperMap_of_comp_of_inj {f : X → Y} {g : Y → Z} (hf : Continuous f) (hg : Continuous g)
    (hgf : IsProperMap (g ∘ f)) (g_inj : g.Injective) : IsProperMap f := by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : X → Y
    g : Y → Z
    hf : Continuous f
    hg : Continuous g
    hgf : IsProperMap (Function.comp g f)
    g_inj : Function.Injective g
    ⊢ IsProperMap f
  -/
  refine ⟨hf, fun ℱ y h ↦ ?_⟩
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : X → Y
    g : Y → Z
    hf : Continuous f
    hg : Continuous g
    hgf : IsProperMap (Function.comp g f)
    g_inj : Function.Injective g
    ℱ : Filter X
    y : Y
    h : MapClusterPt y ℱ f
    ⊢ Exists fun x => And (Eq (f x) y) (ClusterPt x ℱ)
  -/
  rcases hgf.clusterPt_of_mapClusterPt (h.map hg.continuousAt tendsto_map) with ⟨x, hx1, hx2⟩
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : X → Y
    g : Y → Z
    hf : Continuous f
    hg : Continuous g
    hgf : IsProperMap (Function.comp g f)
    g_inj : Function.Injective g
    ℱ : Filter X
    y : Y
    h : MapClusterPt y ℱ f
    x : X
    hx1 : Eq (Function.comp g f x) (g y)
    hx2 : ClusterPt x ℱ
    ⊢ Exists fun x => And (Eq (f x) y) (ClusterPt x ℱ)
  -/
  exact ⟨x, g_inj hx1, hx2⟩
  /-
    🎉 no goals
  -/


/-- If the composition of two continuous functions `f : X → Y` and `g : Y → Z` is proper
and `Y` is T2, then `f` is proper. -/
lemma isProperMap_of_comp_of_t2 [T2Space Y] (hf : Continuous f) (hg : Continuous g)
    (hgf : IsProperMap (g ∘ f)) : IsProperMap f := by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    f : X → Y
    g : Y → Z
    inst✝ : T2Space Y
    hf : Continuous f
    hg : Continuous g
    hgf : IsProperMap (Function.comp g f)
    ⊢ IsProperMap f
  -/
  rw [isProperMap_iff_ultrafilter_of_t2]
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    f : X → Y
    g : Y → Z
    inst✝ : T2Space Y
    hf : Continuous f
    hg : Continuous g
    hgf : IsProperMap (Function.comp g f)
    ⊢ And (Continuous f) (∀ ⦃𝒰 : Ultrafilter X⦄ ⦃y : Y⦄, Filter.Tendsto f (↑𝒰) (nh …
  -/
  refine ⟨hf, fun 𝒰 y h ↦ ?_⟩
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    f : X → Y
    g : Y → Z
    inst✝ : T2Space Y
    hf : Continuous f
    hg : Continuous g
    hgf : IsProperMap (Function.comp g f)
    𝒰 : Ultrafilter X
    y : Y
    h : Filter.Tendsto f (↑𝒰) (nhds y)
    ⊢ Exists fun x => LE.le (↑𝒰) (nhds x)
  -/
  rw [isProperMap_iff_ultrafilter] at hgf
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    f : X → Y
    g : Y → Z
    inst✝ : T2Space Y
    hf : Continuous f
    hg : Continuous g
    hgf : And (Continuous (Function.comp g f)) (∀ ⦃𝒰 : Ultrafilter X⦄ ⦃y : Z⦄, Fil …
    𝒰 : Ultrafilter X
    y : Y
    h : Filter.Tendsto f (↑𝒰) (nhds y)
    ⊢ Exists fun x => LE.le (↑𝒰) (nhds x)
  -/
  rcases hgf.2 ((hg.tendsto y).comp h) with ⟨x, -, hx⟩
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    f : X → Y
    g : Y → Z
    inst✝ : T2Space Y
    hf : Continuous f
    hg : Continuous g
    hgf : And (Continuous (Function.comp g f)) (∀ ⦃𝒰 : Ultrafilter X⦄ ⦃y : Z⦄, Fil …
    𝒰 : Ultrafilter X
    y : Y
    h : Filter.Tendsto f (↑𝒰) (nhds y)
    x : X
    hx : LE.le (↑𝒰) (nhds x)
    ⊢ Exists fun x => LE.le (↑𝒰) (nhds x)
  -/
  exact ⟨x, hx⟩
  /-
    🎉 no goals
  -/


/-- A binary product of proper maps is proper. -/
lemma IsProperMap.prodMap {g : Z → W} (hf : IsProperMap f) (hg : IsProperMap g) :
    IsProperMap (Prod.map f g) := by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    W : Type u_4
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    inst✝ : TopologicalSpace W
    f : X → Y
    g : Z → W
    hf : IsProperMap f
    hg : IsProperMap g
    ⊢ IsProperMap (Prod.map f g)
  -/
  simp_rw [isProperMap_iff_ultrafilter] at hf hg ⊢
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    W : Type u_4
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    inst✝ : TopologicalSpace W
    f : X → Y
    g : Z → W
    hf : And (Continuous f) (∀ ⦃𝒰 : Ultrafilter X⦄ ⦃y : Y⦄, Filter.Tendsto f (↑𝒰)  …
    hg : And (Continuous g) (∀ ⦃𝒰 : Ultrafilter Z⦄ ⦃y : W⦄, Filter.Tendsto g (↑𝒰)  …
    ⊢ And (Continuous (Prod.map f g)) (∀ ⦃𝒰 : Ultrafilter (Prod X Z)⦄ ⦃y : Prod Y  …
  -/
  constructor
  -- Continuity is clear.
    /-
      case left
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      W : Type u_4
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : TopologicalSpace Z
      inst✝ : TopologicalSpace W
      f : X → Y
      g : Z → W
      hf : And (Continuous f) (∀ ⦃𝒰 : Ultrafilter X⦄ ⦃y : Y⦄, Filter.Tendsto f (↑𝒰)  …
      hg : And (Continuous g) (∀ ⦃𝒰 : Ultrafilter Z⦄ ⦃y : W⦄, Filter.Tendsto g (↑𝒰)  …
      ⊢ Continuous (Prod.map f g)
    -/
  · exact hf.1.prodMap hg.1
    /-
      🎉 no goals
    -/
  -- Let `𝒰 : Ultrafilter (X × Z)`, and assume that `f × g` tends to some `(y, w) : Y × W`
  -- along `𝒰`.
    /-
      case right
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      W : Type u_4
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : TopologicalSpace Z
      inst✝ : TopologicalSpace W
      f : X → Y
      g : Z → W
      hf : And (Continuous f) (∀ ⦃𝒰 : Ultrafilter X⦄ ⦃y : Y⦄, Filter.Tendsto f (↑𝒰)  …
      hg : And (Continuous g) (∀ ⦃𝒰 : Ultrafilter Z⦄ ⦃y : W⦄, Filter.Tendsto g (↑𝒰)  …
      ⊢ ∀ ⦃𝒰 : Ultrafilter (Prod X Z)⦄ ⦃y : Prod Y W⦄, Filter.Tendsto (Prod.map f g) …
    -/
  · intro 𝒰 ⟨y, w⟩ hyw
  -- That means that `f` tends to `y` along `map fst 𝒰` and `g` tends to `w` along `map snd 𝒰`.
    /-
      case right
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      W : Type u_4
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : TopologicalSpace Z
      inst✝ : TopologicalSpace W
      f : X → Y
      g : Z → W
      hf : And (Continuous f) (∀ ⦃𝒰 : Ultrafilter X⦄ ⦃y : Y⦄, Filter.Tendsto f (↑𝒰)  …
      hg : And (Continuous g) (∀ ⦃𝒰 : Ultrafilter Z⦄ ⦃y : W⦄, Filter.Tendsto g (↑𝒰)  …
      𝒰 : Ultrafilter (Prod X Z)
      y : Y
      w : W
      hyw : Filter.Tendsto (Prod.map f g) (↑𝒰) (nhds { fst := y, snd := w })
      ⊢ Exists fun x => And (Eq (Prod.map f g x) { fst := y, snd := w }) (LE.le (↑𝒰) …
    -/
    simp_rw [nhds_prod_eq, tendsto_prod_iff'] at hyw
  -- Thus, by properness of `f` and `g`, we get some `x : X` and `z : Z` such that `f x = y`,
  -- `g z = w`, `map fst 𝒰` tends to  `x`, and `map snd 𝒰` tends to `y`.
    rcases hf.2 (show Tendsto f (Ultrafilter.map fst 𝒰) (𝓝 y) by simpa using hyw.1) with
      ⟨x, hxy, hx⟩
    rcases hg.2 (show Tendsto g (Ultrafilter.map snd 𝒰) (𝓝 w) by simpa using hyw.2) with
      ⟨z, hzw, hz⟩
  -- By the properties of the product topology, that means that `𝒰` tends to `(x, z)`,
  -- which completes the proof since `(f × g)(x, z) = (y, w)`.
    /-
      case right.intro.intro.intro.intro
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      W : Type u_4
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : TopologicalSpace Z
      inst✝ : TopologicalSpace W
      f : X → Y
      g : Z → W
      hf : And (Continuous f) (∀ ⦃𝒰 : Ultrafilter X⦄ ⦃y : Y⦄, Filter.Tendsto f (↑𝒰)  …
      hg : And (Continuous g) (∀ ⦃𝒰 : Ultrafilter Z⦄ ⦃y : W⦄, Filter.Tendsto g (↑𝒰)  …
      𝒰 : Ultrafilter (Prod X Z)
      y : Y
      w : W
      hyw : And (Filter.Tendsto (fun n => (Prod.map f g n).1) (↑𝒰) (nhds y)) (Filter …
      x : X
      hxy : Eq (f x) y
      hx : LE.le (↑(Ultrafilter.map Prod.fst 𝒰)) (nhds x)
      z : Z
      hzw : Eq (g z) w
      hz : LE.le (↑(Ultrafilter.map Prod.snd 𝒰)) (nhds z)
      ⊢ Exists fun x => And (Eq (Prod.map f g x) { fst := y, snd := w }) (LE.le (↑𝒰) …
    -/
    refine ⟨⟨x, z⟩, Prod.ext hxy hzw, ?_⟩
    /-
      case right.intro.intro.intro.intro
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      W : Type u_4
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : TopologicalSpace Z
      inst✝ : TopologicalSpace W
      f : X → Y
      g : Z → W
      hf : And (Continuous f) (∀ ⦃𝒰 : Ultrafilter X⦄ ⦃y : Y⦄, Filter.Tendsto f (↑𝒰)  …
      hg : And (Continuous g) (∀ ⦃𝒰 : Ultrafilter Z⦄ ⦃y : W⦄, Filter.Tendsto g (↑𝒰)  …
      𝒰 : Ultrafilter (Prod X Z)
      y : Y
      w : W
      hyw : And (Filter.Tendsto (fun n => (Prod.map f g n).1) (↑𝒰) (nhds y)) (Filter …
      x : X
      hxy : Eq (f x) y
      hx : LE.le (↑(Ultrafilter.map Prod.fst 𝒰)) (nhds x)
      z : Z
      hzw : Eq (g z) w
      hz : LE.le (↑(Ultrafilter.map Prod.snd 𝒰)) (nhds z)
      ⊢ LE.le (↑𝒰) (nhds { fst := x, snd := z })
    -/
    rw [nhds_prod_eq, le_prod]
    /-
      case right.intro.intro.intro.intro
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      W : Type u_4
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : TopologicalSpace Z
      inst✝ : TopologicalSpace W
      f : X → Y
      g : Z → W
      hf : And (Continuous f) (∀ ⦃𝒰 : Ultrafilter X⦄ ⦃y : Y⦄, Filter.Tendsto f (↑𝒰)  …
      hg : And (Continuous g) (∀ ⦃𝒰 : Ultrafilter Z⦄ ⦃y : W⦄, Filter.Tendsto g (↑𝒰)  …
      𝒰 : Ultrafilter (Prod X Z)
      y : Y
      w : W
      hyw : And (Filter.Tendsto (fun n => (Prod.map f g n).1) (↑𝒰) (nhds y)) (Filter …
      x : X
      hxy : Eq (f x) y
      hx : LE.le (↑(Ultrafilter.map Prod.fst 𝒰)) (nhds x)
      z : Z
      hzw : Eq (g z) w
      hz : LE.le (↑(Ultrafilter.map Prod.snd 𝒰)) (nhds z)
      ⊢ And (Filter.Tendsto Prod.fst (↑𝒰) (nhds x)) (Filter.Tendsto Prod.snd (↑𝒰) (n …
    -/
    exact ⟨hx, hz⟩
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-06")] alias IsProperMap.prod_map := IsProperMap.prodMap


/-- Any product of proper maps is proper. -/
lemma IsProperMap.pi_map {X Y : ι → Type*} [∀ i, TopologicalSpace (X i)]
    [∀ i, TopologicalSpace (Y i)] {f : (i : ι) → X i → Y i} (h : ∀ i, IsProperMap (f i)) :
    IsProperMap (fun (x : ∀ i, X i) i ↦ f i (x i)) := by
  /-
    ι : Type u_5
    X : ι → Type u_6
    Y : ι → Type u_7
    inst✝¹ : (i : ι) → TopologicalSpace (X i)
    inst✝ : (i : ι) → TopologicalSpace (Y i)
    f : (i : ι) → X i → Y i
    h : ∀ (i : ι), IsProperMap (f i)
    ⊢ IsProperMap fun x i => f i (x i)
  -/
  simp_rw [isProperMap_iff_ultrafilter] at h ⊢
  /-
    ι : Type u_5
    X : ι → Type u_6
    Y : ι → Type u_7
    inst✝¹ : (i : ι) → TopologicalSpace (X i)
    inst✝ : (i : ι) → TopologicalSpace (Y i)
    f : (i : ι) → X i → Y i
    h : ∀ (i : ι), And (Continuous (f i)) (∀ ⦃𝒰 : Ultrafilter (X i)⦄ ⦃y : Y i⦄, Fi …
    ⊢ And (Continuous fun x i => f i (x i)) (∀ ⦃𝒰 : Ultrafilter ((i : ι) → X i)⦄ ⦃ …
  -/
  constructor
  -- Continuity is clear.
    /-
      case left
      ι : Type u_5
      X : ι → Type u_6
      Y : ι → Type u_7
      inst✝¹ : (i : ι) → TopologicalSpace (X i)
      inst✝ : (i : ι) → TopologicalSpace (Y i)
      f : (i : ι) → X i → Y i
      h : ∀ (i : ι), And (Continuous (f i)) (∀ ⦃𝒰 : Ultrafilter (X i)⦄ ⦃y : Y i⦄, Fi …
      ⊢ Continuous fun x i => f i (x i)
    -/
  · exact continuous_pi fun i ↦ (h i).1.comp (continuous_apply i)
    /-
      🎉 no goals
    -/
  -- Let `𝒰 : Ultrafilter (Π i, X i)`, and assume that `Π i, f i` tends to some `y : Π i, Y i`
  -- along `𝒰`.
    /-
      case right
      ι : Type u_5
      X : ι → Type u_6
      Y : ι → Type u_7
      inst✝¹ : (i : ι) → TopologicalSpace (X i)
      inst✝ : (i : ι) → TopologicalSpace (Y i)
      f : (i : ι) → X i → Y i
      h : ∀ (i : ι), And (Continuous (f i)) (∀ ⦃𝒰 : Ultrafilter (X i)⦄ ⦃y : Y i⦄, Fi …
      ⊢ ∀ ⦃𝒰 : Ultrafilter ((i : ι) → X i)⦄ ⦃y : (i : ι) → Y i⦄, Filter.Tendsto (fun …
    -/
  · intro 𝒰 y hy
  -- That means that each `f i` tends to `y i` along `map (eval i) 𝒰`.
    have : ∀ i, Tendsto (f i) (Ultrafilter.map (eval i) 𝒰) (𝓝 (y i)) := by
      simpa [tendsto_pi_nhds] using hy
  -- Thus, by properness of all the `f i`s, we can choose some `x : Π i, X i` such that, for all
  -- `i`, `f i (x i) = y i` and `map (eval i) 𝒰` tends to  `x i`.
    /-
      case right
      ι : Type u_5
      X : ι → Type u_6
      Y : ι → Type u_7
      inst✝¹ : (i : ι) → TopologicalSpace (X i)
      inst✝ : (i : ι) → TopologicalSpace (Y i)
      f : (i : ι) → X i → Y i
      h : ∀ (i : ι), And (Continuous (f i)) (∀ ⦃𝒰 : Ultrafilter (X i)⦄ ⦃y : Y i⦄, Fi …
      𝒰 : Ultrafilter ((i : ι) → X i)
      y : (i : ι) → Y i
      hy : Filter.Tendsto (fun x i => f i (x i)) (↑𝒰) (nhds y)
      this : ∀ (i : ι), Filter.Tendsto (f i) (↑(Ultrafilter.map (Function.eval i) 𝒰) …
      ⊢ Exists fun x => And (Eq (fun i => f i (x i)) y) (LE.le (↑𝒰) (nhds x))
    -/
    choose x hxy hx using fun i ↦ (h i).2 (this i)
  -- By the properties of the product topology, that means that `𝒰` tends to `x`,
  -- which completes the proof since `(Π i, f i) x = y`.
    /-
      case right
      ι : Type u_5
      X : ι → Type u_6
      Y : ι → Type u_7
      inst✝¹ : (i : ι) → TopologicalSpace (X i)
      inst✝ : (i : ι) → TopologicalSpace (Y i)
      f : (i : ι) → X i → Y i
      h : ∀ (i : ι), And (Continuous (f i)) (∀ ⦃𝒰 : Ultrafilter (X i)⦄ ⦃y : Y i⦄, Fi …
      𝒰 : Ultrafilter ((i : ι) → X i)
      y : (i : ι) → Y i
      hy : Filter.Tendsto (fun x i => f i (x i)) (↑𝒰) (nhds y)
      this : ∀ (i : ι), Filter.Tendsto (f i) (↑(Ultrafilter.map (Function.eval i) 𝒰) …
      x : (i : ι) → X i
      hxy : ∀ (i : ι), Eq (f i (x i)) (y i)
      hx : ∀ (i : ι), LE.le (↑(Ultrafilter.map (Function.eval i) 𝒰)) (nhds (x i))
      ⊢ Exists fun x => And (Eq (fun i => f i (x i)) y) (LE.le (↑𝒰) (nhds x))
    -/
    refine ⟨x, funext hxy, ?_⟩
    /-
      case right
      ι : Type u_5
      X : ι → Type u_6
      Y : ι → Type u_7
      inst✝¹ : (i : ι) → TopologicalSpace (X i)
      inst✝ : (i : ι) → TopologicalSpace (Y i)
      f : (i : ι) → X i → Y i
      h : ∀ (i : ι), And (Continuous (f i)) (∀ ⦃𝒰 : Ultrafilter (X i)⦄ ⦃y : Y i⦄, Fi …
      𝒰 : Ultrafilter ((i : ι) → X i)
      y : (i : ι) → Y i
      hy : Filter.Tendsto (fun x i => f i (x i)) (↑𝒰) (nhds y)
      this : ∀ (i : ι), Filter.Tendsto (f i) (↑(Ultrafilter.map (Function.eval i) 𝒰) …
      x : (i : ι) → X i
      hxy : ∀ (i : ι), Eq (f i (x i)) (y i)
      hx : ∀ (i : ι), LE.le (↑(Ultrafilter.map (Function.eval i) 𝒰)) (nhds (x i))
      ⊢ LE.le (↑𝒰) (nhds x)
    -/
    rwa [nhds_pi, le_pi]
    /-
      🎉 no goals
    -/


/-- The preimage of a compact set by a proper map is again compact. See also
`isProperMap_iff_isCompact_preimage` which proves that this property completely characterizes
proper map when the codomain is compactly generated and Hausdorff. -/
lemma IsProperMap.isCompact_preimage (h : IsProperMap f) {K : Set Y} (hK : IsCompact K) :
    IsCompact (f ⁻¹' K) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    h : IsProperMap f
    K : Set Y
    hK : IsCompact K
    ⊢ IsCompact (Set.preimage f K)
  -/
  rw [isCompact_iff_ultrafilter_le_nhds]
  -- Let `𝒰 ≤ 𝓟 (f ⁻¹' K)` an ultrafilter.
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    h : IsProperMap f
    K : Set Y
    hK : IsCompact K
    ⊢ ∀ (f_1 : Ultrafilter X), LE.le (↑f_1) (Filter.principal (Set.preimage f K))  …
  -/
  intro 𝒰 h𝒰
  -- In other words, we have `map f 𝒰 ≤ 𝓟 K`
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    h : IsProperMap f
    K : Set Y
    hK : IsCompact K
    𝒰 : Ultrafilter X
    h𝒰 : LE.le (↑𝒰) (Filter.principal (Set.preimage f K))
    ⊢ Exists fun x => And (Membership.mem (Set.preimage f K) x) (LE.le (↑𝒰) (nhds  …
  -/
  rw [← comap_principal, ← map_le_iff_le_comap, ← Ultrafilter.coe_map] at h𝒰
  -- Thus, by compactness of `K`, the ultrafilter `map f 𝒰` tends to some `y ∈ K`.
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    h : IsProperMap f
    K : Set Y
    hK : IsCompact K
    𝒰 : Ultrafilter X
    h𝒰 : LE.le (↑(Ultrafilter.map f 𝒰)) (Filter.principal K)
    ⊢ Exists fun x => And (Membership.mem (Set.preimage f K) x) (LE.le (↑𝒰) (nhds  …
  -/
  rcases hK.ultrafilter_le_nhds _ h𝒰 with ⟨y, hyK, hy⟩
  -- Then, by properness of `f`, that means that `𝒰` tends to some `x ∈ f ⁻¹' {y} ⊆ f ⁻¹' K`,
  -- which completes the proof.
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    h : IsProperMap f
    K : Set Y
    hK : IsCompact K
    𝒰 : Ultrafilter X
    h𝒰 : LE.le (↑(Ultrafilter.map f 𝒰)) (Filter.principal K)
    y : Y
    hyK : Membership.mem K y
    hy : LE.le (↑(Ultrafilter.map f 𝒰)) (nhds y)
    ⊢ Exists fun x => And (Membership.mem (Set.preimage f K) x) (LE.le (↑𝒰) (nhds  …
  -/
  rcases h.ultrafilter_le_nhds_of_tendsto hy with ⟨x, rfl, hx⟩
  /-
    case intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    h : IsProperMap f
    K : Set Y
    hK : IsCompact K
    𝒰 : Ultrafilter X
    h𝒰 : LE.le (↑(Ultrafilter.map f 𝒰)) (Filter.principal K)
    x : X
    hx : LE.le (↑𝒰) (nhds x)
    hyK : Membership.mem K (f x)
    hy : LE.le (↑(Ultrafilter.map f 𝒰)) (nhds (f x))
    ⊢ Exists fun x => And (Membership.mem (Set.preimage f K) x) (LE.le (↑𝒰) (nhds  …
  -/
  exact ⟨x, hyK, hx⟩
  /-
    🎉 no goals
  -/


/-- A map is proper if and only if it is closed and its fibers are compact. -/
theorem isProperMap_iff_isClosedMap_and_compact_fibers :
    IsProperMap f ↔ Continuous f ∧ IsClosedMap f ∧ ∀ y, IsCompact (f ⁻¹' {y}) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    ⊢ Iff (IsProperMap f) (And (Continuous f) (And (IsClosedMap f) (∀ (y : Y), IsC …
  -/
  constructor <;> intro H
  -- Note: In Bourbaki, the direct implication is proved by going through universally closed maps.
  -- We could do the same (using a `TFAE` cycle) but proving it directly from
  -- `IsProperMap.isCompact_preimage` is nice enough already so we don't bother with that.
    /-
      case mp
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : IsProperMap f
      ⊢ And (Continuous f) (And (IsClosedMap f) (∀ (y : Y), IsCompact (Set.preimage  …
    -/
  · exact ⟨H.continuous, H.isClosedMap, fun y ↦ H.isCompact_preimage isCompact_singleton⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : And (Continuous f) (And (IsClosedMap f) (∀ (y : Y), IsCompact (Set.preimag …
      ⊢ IsProperMap f
    -/
  · rw [isProperMap_iff_clusterPt]
  -- Let `ℱ : Filter X` and `y` some cluster point of `map f ℱ`.
    /-
      case mpr
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : And (Continuous f) (And (IsClosedMap f) (∀ (y : Y), IsCompact (Set.preimag …
      ⊢ And (Continuous f) (∀ ⦃ℱ : Filter X⦄ ⦃y : Y⦄, MapClusterPt y ℱ f → Exists fu …
    -/
    refine ⟨H.1, fun ℱ y hy ↦ ?_⟩
  -- That means that the singleton `pure y` meets the "closure" of `map f ℱ`, by which we mean
  -- `Filter.lift' (map f ℱ) closure`. But `f` is closed, so
  -- `closure (map f ℱ) = map f (closure ℱ)` (see `IsClosedMap.lift'_closure_map_eq`).
  -- Thus `map f (closure ℱ ⊓ 𝓟 (f ⁻¹' {y})) = map f (closure ℱ) ⊓ 𝓟 {y} ≠ ⊥`, hence
  -- `closure ℱ ⊓ 𝓟 (f ⁻¹' {y}) ≠ ⊥`.
    /-
      case mpr
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : And (Continuous f) (And (IsClosedMap f) (∀ (y : Y), IsCompact (Set.preimag …
      ℱ : Filter X
      y : Y
      hy : MapClusterPt y ℱ f
      ⊢ Exists fun x => And (Eq (f x) y) (ClusterPt x ℱ)
    -/
    rw [H.2.1.mapClusterPt_iff_lift'_closure H.1] at hy
  -- Now, applying the compactness of `f ⁻¹' {y}` to the nontrivial filter
  -- `closure ℱ ⊓ 𝓟 (f ⁻¹' {y})`, we obtain that it has a cluster point `x ∈ f ⁻¹' {y}`.
    /-
      case mpr
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : And (Continuous f) (And (IsClosedMap f) (∀ (y : Y), IsCompact (Set.preimag …
      ℱ : Filter X
      y : Y
      hy : (Min.min (ℱ.lift' closure) (Filter.principal (Set.preimage f (Singleton.s …
      ⊢ Exists fun x => And (Eq (f x) y) (ClusterPt x ℱ)
    -/
    rcases H.2.2 y (f := Filter.lift' ℱ closure ⊓ 𝓟 (f ⁻¹' {y})) inf_le_right with ⟨x, hxy, hx⟩
    /-
      case mpr.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : And (Continuous f) (And (IsClosedMap f) (∀ (y : Y), IsCompact (Set.preimag …
      ℱ : Filter X
      y : Y
      hy : (Min.min (ℱ.lift' closure) (Filter.principal (Set.preimage f (Singleton.s …
      x : X
      hxy : Membership.mem (Set.preimage f (Singleton.singleton y)) x
      hx : ClusterPt x (Min.min (ℱ.lift' closure) (Filter.principal (Set.preimage f  …
      ⊢ Exists fun x => And (Eq (f x) y) (ClusterPt x ℱ)
    -/
    refine ⟨x, hxy, ?_⟩
  -- In particular `x` is a cluster point of `closure ℱ`. Since cluster points of `closure ℱ`
  -- are exactly cluster points of `ℱ` (see `clusterPt_lift'_closure_iff`), this completes
  -- the proof.
    /-
      case mpr.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : And (Continuous f) (And (IsClosedMap f) (∀ (y : Y), IsCompact (Set.preimag …
      ℱ : Filter X
      y : Y
      hy : (Min.min (ℱ.lift' closure) (Filter.principal (Set.preimage f (Singleton.s …
      x : X
      hxy : Membership.mem (Set.preimage f (Singleton.singleton y)) x
      hx : ClusterPt x (Min.min (ℱ.lift' closure) (Filter.principal (Set.preimage f  …
      ⊢ ClusterPt x ℱ
    -/
    rw [← clusterPt_lift'_closure_iff]
    /-
      case mpr.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : And (Continuous f) (And (IsClosedMap f) (∀ (y : Y), IsCompact (Set.preimag …
      ℱ : Filter X
      y : Y
      hy : (Min.min (ℱ.lift' closure) (Filter.principal (Set.preimage f (Singleton.s …
      x : X
      hxy : Membership.mem (Set.preimage f (Singleton.singleton y)) x
      hx : ClusterPt x (Min.min (ℱ.lift' closure) (Filter.principal (Set.preimage f  …
      ⊢ ClusterPt x (ℱ.lift' closure)
    -/
    exact hx.mono inf_le_left
    /-
      🎉 no goals
    -/


/-- An injective and continuous function is proper if and only if it is closed. -/
lemma isProperMap_iff_isClosedMap_of_inj (f_cont : Continuous f) (f_inj : f.Injective) :
    IsProperMap f ↔ IsClosedMap f := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    f_cont : Continuous f
    f_inj : Function.Injective f
    ⊢ Iff (IsProperMap f) (IsClosedMap f)
  -/
  refine ⟨fun h ↦ h.isClosedMap, fun h ↦ ?_⟩
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    f_cont : Continuous f
    f_inj : Function.Injective f
    h : IsClosedMap f
    ⊢ IsProperMap f
  -/
  rw [isProperMap_iff_isClosedMap_and_compact_fibers]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    f_cont : Continuous f
    f_inj : Function.Injective f
    h : IsClosedMap f
    ⊢ And (Continuous f) (And (IsClosedMap f) (∀ (y : Y), IsCompact (Set.preimage  …
  -/
  exact ⟨f_cont, h, fun y ↦ (subsingleton_singleton.preimage f_inj).isCompact⟩
  /-
    🎉 no goals
  -/


/-- A injective continuous and closed map is proper. -/
lemma isProperMap_of_isClosedMap_of_inj (f_cont : Continuous f) (f_inj : f.Injective)
    (f_closed : IsClosedMap f) : IsProperMap f :=
  (isProperMap_iff_isClosedMap_of_inj f_cont f_inj).2 f_closed


/-- A homeomorphism is proper. -/
@[simp] lemma Homeomorph.isProperMap (e : X ≃ₜ Y) : IsProperMap e :=
  isProperMap_of_isClosedMap_of_inj e.continuous e.injective e.isClosedMap


protected lemma IsHomeomorph.isProperMap (hf : IsHomeomorph f) : IsProperMap f :=
  isProperMap_of_isClosedMap_of_inj hf.continuous hf.injective hf.isClosedMap


/-- The identity is proper. -/
@[simp] lemma isProperMap_id : IsProperMap (id : X → X) := IsHomeomorph.id.isProperMap


/-- A closed embedding is proper. -/
lemma Topology.IsClosedEmbedding.isProperMap (hf : IsClosedEmbedding f) : IsProperMap f :=
  isProperMap_of_isClosedMap_of_inj hf.continuous hf.injective hf.isClosedMap


@[deprecated (since := "2024-10-20")]
alias isProperMap_of_closedEmbedding := IsClosedEmbedding.isProperMap


/-- The coercion from a closed subset is proper. -/
lemma IsClosed.isProperMap_subtypeVal {C : Set X} (hC : IsClosed C) : IsProperMap ((↑) : C → X) :=
  hC.isClosedEmbedding_subtypeVal.isProperMap


@[deprecated (since := "2024-10-20")]
alias isProperMap_subtype_val_of_closed := IsClosed.isProperMap_subtypeVal


/-- The restriction of a proper map to a closed subset is proper. -/
lemma IsProperMap.restrict {C : Set X} (hf : IsProperMap f) (hC : IsClosed C) :
    IsProperMap fun x : C ↦ f x := hC.isProperMap_subtypeVal.comp  hf


@[deprecated (since := "2024-10-20")]
alias isProperMap_restr_of_proper_of_closed := IsProperMap.restrict


/-- The range of a proper map is closed. -/
lemma IsProperMap.isClosed_range (hf : IsProperMap f) : IsClosed (range f) :=
  hf.isClosedMap.isClosed_range


@[deprecated (since := "2024-05-08")] alias IsProperMap.closed_range := IsProperMap.isClosed_range


/-- Version of `isProperMap_iff_isClosedMap_and_compact_fibers` in terms of `cofinite` and
`cocompact`. Only works when the codomain is `T1`. -/
lemma isProperMap_iff_isClosedMap_and_tendsto_cofinite [T1Space Y] :
    IsProperMap f ↔ Continuous f ∧ IsClosedMap f ∧ Tendsto f (cocompact X) cofinite := by
  simp_rw [isProperMap_iff_isClosedMap_and_compact_fibers, Tendsto,
    le_cofinite_iff_compl_singleton_mem, mem_map, preimage_compl]
  refine and_congr_right fun f_cont ↦ and_congr_right fun _ ↦
    ⟨fun H y ↦ (H y).compl_mem_cocompact, fun H y ↦ ?_⟩
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    inst✝ : T1Space Y
    f_cont : Continuous f
    x✝ : IsClosedMap f
    H : ∀ (x : Y), Membership.mem (Filter.cocompact X) (HasCompl.compl (Set.preima …
    y : Y
    ⊢ IsCompact (Set.preimage f (Singleton.singleton y))
  -/
  rcases mem_cocompact.mp (H y) with ⟨K, hK, hKy⟩
  exact hK.of_isClosed_subset (isClosed_singleton.preimage f_cont)
    (compl_le_compl_iff_le.mp hKy)


/-- A continuous map from a compact space to a T₂ space is a proper map. -/
theorem Continuous.isProperMap [CompactSpace X] [T2Space Y] (hf : Continuous f) : IsProperMap f :=
                                                                             /-
                                                                               X : Type u_1
                                                                               Y : Type u_2
                                                                               inst✝³ : TopologicalSpace X
                                                                               inst✝² : TopologicalSpace Y
                                                                               f : X → Y
                                                                               inst✝¹ : CompactSpace X
                                                                               inst✝ : T2Space Y
                                                                               hf : Continuous f
                                                                               ⊢ Filter.Tendsto f (Filter.cocompact X) Filter.cofinite
                                                                             -/
  isProperMap_iff_isClosedMap_and_tendsto_cofinite.2 ⟨hf, hf.isClosedMap, by simp⟩
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


/-- A proper map `f : X → Y` is **universally closed**: for any topological space `Z`, the map
`Prod.map f id : X × Z → Y × Z` is closed. We will prove in `isProperMap_iff_universally_closed`
that proper maps are exactly continuous maps which have this property, but this result should be
easier to use because it allows `Z` to live in any universe. -/
theorem IsProperMap.universally_closed (Z) [TopologicalSpace Z] (h : IsProperMap f) :
    IsClosedMap (Prod.map f id : X × Z → Y × Z) :=
  -- `f × id` is proper as a product of proper maps, hence closed.
  (h.prodMap isProperMap_id).isClosedMap


/-- A map `f : X → Y` is proper if and only if it is continuous and the map
`(Prod.map f id : X × Filter X → Y × Filter X)` is closed. This is stronger than
`isProperMap_iff_universally_closed` since it shows that there's only one space to check to get
properness, but in most cases it doesn't matter. -/
theorem isProperMap_iff_isClosedMap_filter {X : Type u} {Y : Type v} [TopologicalSpace X]
    [TopologicalSpace Y] {f : X → Y} :
    IsProperMap f ↔ Continuous f ∧ IsClosedMap
      (Prod.map f id : X × Filter X → Y × Filter X) := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    ⊢ Iff (IsProperMap f) (And (Continuous f) (IsClosedMap (Prod.map f id)))
  -/
  constructor <;> intro H
  -- The direct implication is clear.
    /-
      case mp
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : IsProperMap f
      ⊢ And (Continuous f) (IsClosedMap (Prod.map f id))
    -/
  · exact ⟨H.continuous, H.universally_closed _⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : And (Continuous f) (IsClosedMap (Prod.map f id))
      ⊢ IsProperMap f
    -/
  · rw [isProperMap_iff_ultrafilter]
  -- Let `𝒰 : Ultrafilter X`, and assume that `f` tends to some `y` along `𝒰`.
    /-
      case mpr
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : And (Continuous f) (IsClosedMap (Prod.map f id))
      ⊢ And (Continuous f) (∀ ⦃𝒰 : Ultrafilter X⦄ ⦃y : Y⦄, Filter.Tendsto f (↑𝒰) (nh …
    -/
    refine ⟨H.1, fun 𝒰 y hy ↦ ?_⟩
  -- In `X × Filter X`, consider the closed set `F := closure {(x, ℱ) | ℱ = pure x}`
    /-
      case mpr
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : And (Continuous f) (IsClosedMap (Prod.map f id))
      𝒰 : Ultrafilter X
      y : Y
      hy : Filter.Tendsto f (↑𝒰) (nhds y)
      ⊢ Exists fun x => And (Eq (f x) y) (LE.le (↑𝒰) (nhds x))
    -/
    let F : Set (X × Filter X) := closure {xℱ | xℱ.2 = pure xℱ.1}
  -- Since `f × id` is closed, the set `(f × id) '' F` is also closed.
    /-
      case mpr
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : And (Continuous f) (IsClosedMap (Prod.map f id))
      𝒰 : Ultrafilter X
      y : Y
      hy : Filter.Tendsto f (↑𝒰) (nhds y)
      F : Set (Prod X (Filter X)) := closure (setOf fun xℱ => Eq xℱ.2 (Pure.pure xℱ. …
      ⊢ Exists fun x => And (Eq (f x) y) (LE.le (↑𝒰) (nhds x))
    -/
    have := H.2 F isClosed_closure
  -- Let us show that `(y, 𝒰) ∈ (f × id) '' F`.
    have : (y, ↑𝒰) ∈ Prod.map f id '' F :=
  -- Note that, by the properties of the topology on `Filter X`, the function `pure : X → Filter X`
  -- tends to the point `𝒰` of `Filter X` along the filter `𝒰` on `X`. Since `f` tends to `y` along
  -- `𝒰`, we get that the function `(f, pure) : X → (Y, Filter X)` tends to `(y, 𝒰)` along
  -- `𝒰`. Furthermore, each `(f, pure)(x) = (f × id)(x, pure x)` is clearly an element of
  -- the closed set `(f × id) '' F`, thus the limit `(y, 𝒰)` also belongs to that set.
      this.mem_of_tendsto (hy.prod_mk_nhds (Filter.tendsto_pure_self (𝒰 : Filter X)))
        (Eventually.of_forall fun x ↦ ⟨⟨x, pure x⟩, subset_closure rfl, rfl⟩)
  -- The above shows that `(y, 𝒰) = (f x, 𝒰)`, for some `x : X` such that `(x, 𝒰) ∈ F`.
    /-
      case mpr
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : And (Continuous f) (IsClosedMap (Prod.map f id))
      𝒰 : Ultrafilter X
      y : Y
      hy : Filter.Tendsto f (↑𝒰) (nhds y)
      F : Set (Prod X (Filter X)) := closure (setOf fun xℱ => Eq xℱ.2 (Pure.pure xℱ. …
      this✝ : IsClosed (Set.image (Prod.map f id) F)
      this : Membership.mem (Set.image (Prod.map f id) F) { fst := y, snd := ↑𝒰 }
      ⊢ Exists fun x => And (Eq (f x) y) (LE.le (↑𝒰) (nhds x))
    -/
    rcases this with ⟨⟨x, _⟩, hx, ⟨_, _⟩⟩
  -- We already know that `f x = y`, so to finish the proof we just have to check that `𝒰` tends
  -- to `x`. So, for `U ∈ 𝓝 x` arbitrary, let's show that `U ∈ 𝒰`. Since `𝒰` is a ultrafilter,
  -- it is enough to show that `Uᶜ` is not in `𝒰`.
    /-
      case mpr.intro.mk.intro.refl
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : And (Continuous f) (IsClosedMap (Prod.map f id))
      𝒰 : Ultrafilter X
      F : Set (Prod X (Filter X)) := closure (setOf fun xℱ => Eq xℱ.2 (Pure.pure xℱ. …
      this : IsClosed (Set.image (Prod.map f id) F)
      x : X
      hy : Filter.Tendsto f (↑𝒰) (nhds (f x))
      hx : Membership.mem F { fst := x, snd := 𝒰.1 }
      ⊢ Exists fun x_1 => And (Eq (f x_1) (f x)) (LE.le (↑𝒰) (nhds x_1))
    -/
    refine ⟨x, rfl, fun U hU ↦ Ultrafilter.compl_not_mem_iff.mp fun hUc ↦ ?_⟩
    /-
      case mpr.intro.mk.intro.refl
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : And (Continuous f) (IsClosedMap (Prod.map f id))
      𝒰 : Ultrafilter X
      F : Set (Prod X (Filter X)) := closure (setOf fun xℱ => Eq xℱ.2 (Pure.pure xℱ. …
      this : IsClosed (Set.image (Prod.map f id) F)
      x : X
      hy : Filter.Tendsto f (↑𝒰) (nhds (f x))
      hx : Membership.mem F { fst := x, snd := 𝒰.1 }
      U : Set X
      hU : Membership.mem (nhds x) U
      hUc : Membership.mem 𝒰 (HasCompl.compl U)
      ⊢ False
    -/
    rw [mem_closure_iff_nhds] at hx
  -- Indeed, if that was the case, the set `V := {𝒢 : Filter X | Uᶜ ∈ 𝒢}` would be a neighborhood
  -- of `𝒰` in `Filter X`, hence `U ×ˢ V` would be a neighborhood of `(x, 𝒰) : X × Filter X`.
  -- But recall that `(x, 𝒰) ∈ F = closure {(x, ℱ) | ℱ = pure x}`, so the neighborhood `U ×ˢ V`
  -- must contain some element of the form `(z, pure z)`. In other words, we have `z ∈ U` and
  -- `Uᶜ ∈ pure z`, which means `z ∈ Uᶜ` by the definition of pure.
  -- This is a contradiction, which completes the proof.
    rcases hx (U ×ˢ {𝒢 | Uᶜ ∈ 𝒢}) (prod_mem_nhds hU (isOpen_setOf_mem.mem_nhds hUc)) with
      ⟨⟨z, 𝒢⟩, ⟨⟨hz : z ∈ U, hz' : Uᶜ ∈ 𝒢⟩, rfl : 𝒢 = pure z⟩⟩
    /-
      case mpr.intro.mk.intro.refl.intro.mk.intro.intro
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : And (Continuous f) (IsClosedMap (Prod.map f id))
      𝒰 : Ultrafilter X
      F : Set (Prod X (Filter X)) := closure (setOf fun xℱ => Eq xℱ.2 (Pure.pure xℱ. …
      this : IsClosed (Set.image (Prod.map f id) F)
      x : X
      hy : Filter.Tendsto f (↑𝒰) (nhds (f x))
      hx : ∀ (t : Set (Prod X (Filter X))), Membership.mem (nhds { fst := x, snd :=  …
      U : Set X
      hU : Membership.mem (nhds x) U
      hUc : Membership.mem 𝒰 (HasCompl.compl U)
      z : X
      hz : Membership.mem U z
      hz' : Membership.mem (Pure.pure z) (HasCompl.compl U)
      ⊢ False
    -/
    exact hz' hz
    /-
      🎉 no goals
    -/

