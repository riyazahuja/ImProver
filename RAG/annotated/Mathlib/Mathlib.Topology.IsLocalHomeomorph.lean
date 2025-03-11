/-- A function `f : X → Y` satisfies `IsLocalHomeomorphOn f s` if each `x ∈ s` is contained in
the source of some `e : PartialHomeomorph X Y` with `f = e`. -/
def IsLocalHomeomorphOn :=
  ∀ x ∈ s, ∃ e : PartialHomeomorph X Y, x ∈ e.source ∧ f = e


theorem isLocalHomeomorphOn_iff_isOpenEmbedding_restrict {f : X → Y} :
    IsLocalHomeomorphOn f s ↔ ∀ x ∈ s, ∃ U ∈ 𝓝 x, IsOpenEmbedding (U.restrict f) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    f : X → Y
    ⊢ Iff (IsLocalHomeomorphOn f s) (∀ (x : X), Membership.mem s x → Exists fun U  …
  -/
  refine ⟨fun h x hx ↦ ?_, fun h x hx ↦ ?_⟩
    /-
      case refine_1
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      s : Set X
      f : X → Y
      h : IsLocalHomeomorphOn f s
      x : X
      hx : Membership.mem s x
      ⊢ Exists fun U => And (Membership.mem (nhds x) U) (Topology.IsOpenEmbedding (U …
    -/
  · obtain ⟨e, hxe, rfl⟩ := h x hx
    /-
      case refine_1.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      s : Set X
      x : X
      hx : Membership.mem s x
      e : PartialHomeomorph X Y
      hxe : Membership.mem e.source x
      h : IsLocalHomeomorphOn (↑e) s
      ⊢ Exists fun U => And (Membership.mem (nhds x) U) (Topology.IsOpenEmbedding (U …
    -/
    exact ⟨e.source, e.open_source.mem_nhds hxe, e.isOpenEmbedding_restrict⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      s : Set X
      f : X → Y
      h : ∀ (x : X), Membership.mem s x → Exists fun U => And (Membership.mem (nhds  …
      x : X
      hx : Membership.mem s x
      ⊢ Exists fun e => And (Membership.mem e.source x) (Eq f ↑e)
    -/
  · obtain ⟨U, hU, emb⟩ := h x hx
    have : IsOpenEmbedding ((interior U).restrict f) := by
      refine emb.comp ⟨.inclusion interior_subset, ?_⟩
      rw [Set.range_inclusion]; exact isOpen_induced isOpen_interior
    /-
      case refine_2.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      s : Set X
      f : X → Y
      h : ∀ (x : X), Membership.mem s x → Exists fun U => And (Membership.mem (nhds  …
      x : X
      hx : Membership.mem s x
      U : Set X
      hU : Membership.mem (nhds x) U
      emb : Topology.IsOpenEmbedding (U.restrict f)
      this : Topology.IsOpenEmbedding ((interior U).restrict f)
      ⊢ Exists fun e => And (Membership.mem e.source x) (Eq f ↑e)
    -/
    obtain ⟨cont, inj, openMap⟩ := isOpenEmbedding_iff_continuous_injective_isOpenMap.mp this
    /-
      case refine_2.intro.intro.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      s : Set X
      f : X → Y
      h : ∀ (x : X), Membership.mem s x → Exists fun U => And (Membership.mem (nhds  …
      x : X
      hx : Membership.mem s x
      U : Set X
      hU : Membership.mem (nhds x) U
      emb : Topology.IsOpenEmbedding (U.restrict f)
      this : Topology.IsOpenEmbedding ((interior U).restrict f)
      cont : Continuous ((interior U).restrict f)
      inj : Function.Injective ((interior U).restrict f)
      openMap : IsOpenMap ((interior U).restrict f)
      ⊢ Exists fun e => And (Membership.mem e.source x) (Eq f ↑e)
    -/
    haveI : Nonempty X := ⟨x⟩
    exact ⟨PartialHomeomorph.ofContinuousOpenRestrict
      (Set.injOn_iff_injective.mpr inj).toPartialEquiv
      (continuousOn_iff_continuous_restrict.mpr cont) openMap isOpen_interior,
      mem_interior_iff_mem_nhds.mpr hU, rfl⟩


@[deprecated (since := "2024-10-18")]
alias isLocalHomeomorphOn_iff_openEmbedding_restrict :=
  isLocalHomeomorphOn_iff_isOpenEmbedding_restrict


/-- Proves that `f` satisfies `IsLocalHomeomorphOn f s`. The condition `h` is weaker than the
definition of `IsLocalHomeomorphOn f s`, since it only requires `e : PartialHomeomorph X Y` to
agree with `f` on its source `e.source`, as opposed to on the whole space `X`. -/
theorem mk (h : ∀ x ∈ s, ∃ e : PartialHomeomorph X Y, x ∈ e.source ∧ Set.EqOn f e e.source) :
    IsLocalHomeomorphOn f s := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    s : Set X
    h : ∀ (x : X), Membership.mem s x → Exists fun e => And (Membership.mem e.sour …
    ⊢ IsLocalHomeomorphOn f s
  -/
  intro x hx
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    s : Set X
    h : ∀ (x : X), Membership.mem s x → Exists fun e => And (Membership.mem e.sour …
    x : X
    hx : Membership.mem s x
    ⊢ Exists fun e => And (Membership.mem e.source x) (Eq f ↑e)
  -/
  obtain ⟨e, hx, he⟩ := h x hx
  exact
    ⟨{ e with
        toFun := f
        map_source' := fun _x hx ↦ by rw [he hx]; exact e.map_source' hx
        left_inv' := fun _x hx ↦ by rw [he hx]; exact e.left_inv' hx
        right_inv' := fun _y hy ↦ by rw [he (e.map_target' hy)]; exact e.right_inv' hy
        continuousOn_toFun := (continuousOn_congr he).mpr e.continuousOn_toFun },
      hx, rfl⟩


/-- A `PartialHomeomorph` is a local homeomorphism on its source. -/
lemma PartialHomeomorph.isLocalHomeomorphOn (e : PartialHomeomorph X Y) :
    IsLocalHomeomorphOn e e.source :=
  fun _ hx ↦ ⟨e, hx, rfl⟩


theorem mono {t : Set X} (hf : IsLocalHomeomorphOn f t) (hst : s ⊆ t) : IsLocalHomeomorphOn f s :=
  fun x hx ↦ hf x (hst hx)


theorem of_comp_left (hgf : IsLocalHomeomorphOn (g ∘ f) s) (hg : IsLocalHomeomorphOn g (f '' s))
    (cont : ∀ x ∈ s, ContinuousAt f x) : IsLocalHomeomorphOn f s := mk f s fun x hx ↦ by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    g : Y → Z
    f : X → Y
    s : Set X
    hgf : IsLocalHomeomorphOn (Function.comp g f) s
    hg : IsLocalHomeomorphOn g (Set.image f s)
    cont : ∀ (x : X), Membership.mem s x → ContinuousAt f x
    x : X
    hx : Membership.mem s x
    ⊢ Exists fun e => And (Membership.mem e.source x) (Set.EqOn f (↑e) e.source)
  -/
  obtain ⟨g, hxg, rfl⟩ := hg (f x) ⟨x, hx, rfl⟩
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : X → Y
    s : Set X
    cont : ∀ (x : X), Membership.mem s x → ContinuousAt f x
    x : X
    hx : Membership.mem s x
    g : PartialHomeomorph Y Z
    hxg : Membership.mem g.source (f x)
    hgf : IsLocalHomeomorphOn (Function.comp (↑g) f) s
    hg : IsLocalHomeomorphOn (↑g) (Set.image f s)
    ⊢ Exists fun e => And (Membership.mem e.source x) (Set.EqOn f (↑e) e.source)
  -/
  obtain ⟨gf, hgf, he⟩ := hgf x hx
  refine ⟨(gf.restr <| f ⁻¹' g.source).trans g.symm, ⟨⟨hgf, mem_interior_iff_mem_nhds.mpr
    ((cont x hx).preimage_mem_nhds <| g.open_source.mem_nhds hxg)⟩, he ▸ g.map_source hxg⟩,
    fun y hy ↦ ?_⟩
  /-
    case intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : X → Y
    s : Set X
    cont : ∀ (x : X), Membership.mem s x → ContinuousAt f x
    x : X
    hx : Membership.mem s x
    g : PartialHomeomorph Y Z
    hxg : Membership.mem g.source (f x)
    hgf✝ : IsLocalHomeomorphOn (Function.comp (↑g) f) s
    hg : IsLocalHomeomorphOn (↑g) (Set.image f s)
    gf : PartialHomeomorph X Z
    hgf : Membership.mem gf.source x
    he : Eq (Function.comp (↑g) f) ↑gf
    y : X
    hy : Membership.mem ((gf.restr (Set.preimage f g.source)).trans g.symm).source y
    ⊢ Eq (f y) (↑((gf.restr (Set.preimage f g.source)).trans g.symm) y)
  -/
  change f y = g.symm (gf y)
  /-
    case intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : X → Y
    s : Set X
    cont : ∀ (x : X), Membership.mem s x → ContinuousAt f x
    x : X
    hx : Membership.mem s x
    g : PartialHomeomorph Y Z
    hxg : Membership.mem g.source (f x)
    hgf✝ : IsLocalHomeomorphOn (Function.comp (↑g) f) s
    hg : IsLocalHomeomorphOn (↑g) (Set.image f s)
    gf : PartialHomeomorph X Z
    hgf : Membership.mem gf.source x
    he : Eq (Function.comp (↑g) f) ↑gf
    y : X
    hy : Membership.mem ((gf.restr (Set.preimage f g.source)).trans g.symm).source y
    ⊢ Eq (f y) (↑g.symm (↑gf y))
  -/
  have : f y ∈ g.source := by apply interior_subset hy.1.2
  /-
    case intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : X → Y
    s : Set X
    cont : ∀ (x : X), Membership.mem s x → ContinuousAt f x
    x : X
    hx : Membership.mem s x
    g : PartialHomeomorph Y Z
    hxg : Membership.mem g.source (f x)
    hgf✝ : IsLocalHomeomorphOn (Function.comp (↑g) f) s
    hg : IsLocalHomeomorphOn (↑g) (Set.image f s)
    gf : PartialHomeomorph X Z
    hgf : Membership.mem gf.source x
    he : Eq (Function.comp (↑g) f) ↑gf
    y : X
    hy : Membership.mem ((gf.restr (Set.preimage f g.source)).trans g.symm).source y
    this : Membership.mem g.source (f y)
    ⊢ Eq (f y) (↑g.symm (↑gf y))
  -/
  rw [← he, g.eq_symm_apply this (by apply g.map_source this), Function.comp_apply]
  /-
    🎉 no goals
  -/


theorem of_comp_right (hgf : IsLocalHomeomorphOn (g ∘ f) s) (hf : IsLocalHomeomorphOn f s) :
    IsLocalHomeomorphOn g (f '' s) := mk g _ <| by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    g : Y → Z
    f : X → Y
    s : Set X
    hgf : IsLocalHomeomorphOn (Function.comp g f) s
    hf : IsLocalHomeomorphOn f s
    ⊢ ∀ (x : Y), Membership.mem (Set.image f s) x → Exists fun e => And (Membershi …
  -/
  rintro _ ⟨x, hx, rfl⟩
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    g : Y → Z
    f : X → Y
    s : Set X
    hgf : IsLocalHomeomorphOn (Function.comp g f) s
    hf : IsLocalHomeomorphOn f s
    x : X
    hx : Membership.mem s x
    ⊢ Exists fun e => And (Membership.mem e.source (f x)) (Set.EqOn g (↑e) e.source)
  -/
  obtain ⟨f, hxf, rfl⟩ := hf x hx
  /-
    case intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    g : Y → Z
    s : Set X
    x : X
    hx : Membership.mem s x
    f : PartialHomeomorph X Y
    hxf : Membership.mem f.source x
    hgf : IsLocalHomeomorphOn (Function.comp g ↑f) s
    hf : IsLocalHomeomorphOn (↑f) s
    ⊢ Exists fun e => And (Membership.mem e.source (↑f x)) (Set.EqOn g (↑e) e.sour …
  -/
  obtain ⟨gf, hgf, he⟩ := hgf x hx
  /-
    case intro.intro.intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    g : Y → Z
    s : Set X
    x : X
    hx : Membership.mem s x
    f : PartialHomeomorph X Y
    hxf : Membership.mem f.source x
    hgf✝ : IsLocalHomeomorphOn (Function.comp g ↑f) s
    hf : IsLocalHomeomorphOn (↑f) s
    gf : PartialHomeomorph X Z
    hgf : Membership.mem gf.source x
    he : Eq (Function.comp g ↑f) ↑gf
    ⊢ Exists fun e => And (Membership.mem e.source (↑f x)) (Set.EqOn g (↑e) e.sour …
  -/
  refine ⟨f.symm.trans gf, ⟨f.map_source hxf, ?_⟩, fun y hy ↦ ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.refine_1
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      g : Y → Z
      s : Set X
      x : X
      hx : Membership.mem s x
      f : PartialHomeomorph X Y
      hxf : Membership.mem f.source x
      hgf✝ : IsLocalHomeomorphOn (Function.comp g ↑f) s
      hf : IsLocalHomeomorphOn (↑f) s
      gf : PartialHomeomorph X Z
      hgf : Membership.mem gf.source x
      he : Eq (Function.comp g ↑f) ↑gf
      ⊢ Membership.mem (Set.preimage (↑f.symm.symm.symm) gf.source) (↑f x)
    -/
  · apply (f.left_inv hxf).symm ▸ hgf
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.refine_2
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      g : Y → Z
      s : Set X
      x : X
      hx : Membership.mem s x
      f : PartialHomeomorph X Y
      hxf : Membership.mem f.source x
      hgf✝ : IsLocalHomeomorphOn (Function.comp g ↑f) s
      hf : IsLocalHomeomorphOn (↑f) s
      gf : PartialHomeomorph X Z
      hgf : Membership.mem gf.source x
      he : Eq (Function.comp g ↑f) ↑gf
      y : Y
      hy : Membership.mem (f.symm.trans gf).source y
      ⊢ Eq (g y) (↑(f.symm.trans gf) y)
    -/
  · change g y = gf (f.symm y)
    /-
      case intro.intro.intro.intro.intro.intro.refine_2
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      g : Y → Z
      s : Set X
      x : X
      hx : Membership.mem s x
      f : PartialHomeomorph X Y
      hxf : Membership.mem f.source x
      hgf✝ : IsLocalHomeomorphOn (Function.comp g ↑f) s
      hf : IsLocalHomeomorphOn (↑f) s
      gf : PartialHomeomorph X Z
      hgf : Membership.mem gf.source x
      he : Eq (Function.comp g ↑f) ↑gf
      y : Y
      hy : Membership.mem (f.symm.trans gf).source y
      ⊢ Eq (g y) (↑gf (↑f.symm y))
    -/
    rw [← he, Function.comp_apply, f.right_inv hy.1]
    /-
      🎉 no goals
    -/


theorem map_nhds_eq (hf : IsLocalHomeomorphOn f s) {x : X} (hx : x ∈ s) : (𝓝 x).map f = 𝓝 (f x) :=
  let ⟨e, hx, he⟩ := hf x hx
  he.symm ▸ e.map_nhds_eq hx


protected theorem continuousAt (hf : IsLocalHomeomorphOn f s) {x : X} (hx : x ∈ s) :
    ContinuousAt f x :=
  (hf.map_nhds_eq hx).le


protected theorem continuousOn (hf : IsLocalHomeomorphOn f s) : ContinuousOn f s :=
  continuousOn_of_forall_continuousAt fun _x ↦ hf.continuousAt


protected theorem comp (hg : IsLocalHomeomorphOn g t) (hf : IsLocalHomeomorphOn f s)
    (h : Set.MapsTo f s t) : IsLocalHomeomorphOn (g ∘ f) s := by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    g : Y → Z
    f : X → Y
    s : Set X
    t : Set Y
    hg : IsLocalHomeomorphOn g t
    hf : IsLocalHomeomorphOn f s
    h : Set.MapsTo f s t
    ⊢ IsLocalHomeomorphOn (Function.comp g f) s
  -/
  intro x hx
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    g : Y → Z
    f : X → Y
    s : Set X
    t : Set Y
    hg : IsLocalHomeomorphOn g t
    hf : IsLocalHomeomorphOn f s
    h : Set.MapsTo f s t
    x : X
    hx : Membership.mem s x
    ⊢ Exists fun e => And (Membership.mem e.source x) (Eq (Function.comp g f) ↑e)
  -/
  obtain ⟨eg, hxg, rfl⟩ := hg (f x) (h hx)
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : X → Y
    s : Set X
    t : Set Y
    hf : IsLocalHomeomorphOn f s
    h : Set.MapsTo f s t
    x : X
    hx : Membership.mem s x
    eg : PartialHomeomorph Y Z
    hxg : Membership.mem eg.source (f x)
    hg : IsLocalHomeomorphOn (↑eg) t
    ⊢ Exists fun e => And (Membership.mem e.source x) (Eq (Function.comp (↑eg) f)  …
  -/
  obtain ⟨ef, hxf, rfl⟩ := hf x hx
  /-
    case intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    s : Set X
    t : Set Y
    x : X
    hx : Membership.mem s x
    eg : PartialHomeomorph Y Z
    hg : IsLocalHomeomorphOn (↑eg) t
    ef : PartialHomeomorph X Y
    hxf : Membership.mem ef.source x
    hf : IsLocalHomeomorphOn (↑ef) s
    h : Set.MapsTo (↑ef) s t
    hxg : Membership.mem eg.source (↑ef x)
    ⊢ Exists fun e => And (Membership.mem e.source x) (Eq (Function.comp ↑eg ↑ef)  …
  -/
  exact ⟨ef.trans eg, ⟨hxf, hxg⟩, rfl⟩
  /-
    🎉 no goals
  -/


/-- A function `f : X → Y` satisfies `IsLocalHomeomorph f` if each `x : x` is contained in
  the source of some `e : PartialHomeomorph X Y` with `f = e`. -/
def IsLocalHomeomorph :=
  ∀ x : X, ∃ e : PartialHomeomorph X Y, x ∈ e.source ∧ f = e


theorem Homeomorph.isLocalHomeomorph (f : X ≃ₜ Y) : IsLocalHomeomorph f :=
  fun _ ↦ ⟨f.toPartialHomeomorph, trivial, rfl⟩


theorem isLocalHomeomorph_iff_isLocalHomeomorphOn_univ :
    IsLocalHomeomorph f ↔ IsLocalHomeomorphOn f Set.univ :=
  ⟨fun h x _ ↦ h x, fun h x ↦ h x trivial⟩


protected theorem IsLocalHomeomorph.isLocalHomeomorphOn (hf : IsLocalHomeomorph f) :
    IsLocalHomeomorphOn f s := fun x _ ↦ hf x


theorem isLocalHomeomorph_iff_isOpenEmbedding_restrict {f : X → Y} :
    IsLocalHomeomorph f ↔ ∀ x : X, ∃ U ∈ 𝓝 x, IsOpenEmbedding (U.restrict f) := by
  simp_rw [isLocalHomeomorph_iff_isLocalHomeomorphOn_univ,
    isLocalHomeomorphOn_iff_isOpenEmbedding_restrict, imp_iff_right (Set.mem_univ _)]


@[deprecated (since := "2024-10-18")]
alias isLocalHomeomorph_iff_openEmbedding_restrict := isLocalHomeomorph_iff_isOpenEmbedding_restrict


theorem Topology.IsOpenEmbedding.isLocalHomeomorph (hf : IsOpenEmbedding f) : IsLocalHomeomorph f :=
  isLocalHomeomorph_iff_isOpenEmbedding_restrict.mpr fun _ ↦
    ⟨_, Filter.univ_mem, hf.comp (Homeomorph.Set.univ X).isOpenEmbedding⟩


@[deprecated (since := "2024-10-18")]
alias OpenEmbedding.isLocalHomeomorph := IsOpenEmbedding.isLocalHomeomorph


/-- Proves that `f` satisfies `IsLocalHomeomorph f`. The condition `h` is weaker than the
definition of `IsLocalHomeomorph f`, since it only requires `e : PartialHomeomorph X Y` to
agree with `f` on its source `e.source`, as opposed to on the whole space `X`. -/
theorem mk (h : ∀ x : X, ∃ e : PartialHomeomorph X Y, x ∈ e.source ∧ Set.EqOn f e e.source) :
    IsLocalHomeomorph f :=
  isLocalHomeomorph_iff_isLocalHomeomorphOn_univ.mpr
    (IsLocalHomeomorphOn.mk f Set.univ fun x _hx ↦ h x)


/-- A homeomorphism is a local homeomorphism. -/
lemma Homeomorph.isLocalHomeomorph (h : X ≃ₜ Y) : IsLocalHomeomorph h :=
  fun _ ↦ ⟨h.toPartialHomeomorph, trivial, rfl⟩


lemma isLocallyInjective (hf : IsLocalHomeomorph f) : IsLocallyInjective f :=
             /-
               X : Type u_1
               Y : Type u_2
               inst✝¹ : TopologicalSpace X
               inst✝ : TopologicalSpace Y
               f : X → Y
               hf : IsLocalHomeomorph f
               x : X
               ⊢ Exists fun U => And (IsOpen U) (And (Membership.mem U x) (Set.InjOn f U))
             -/
  fun x ↦ by obtain ⟨f, hx, rfl⟩ := hf x; exact ⟨f.source, f.open_source, hx, f.injOn⟩
                                          /-
                                            🎉 no goals
                                          -/


theorem of_comp (hgf : IsLocalHomeomorph (g ∘ f)) (hg : IsLocalHomeomorph g)
    (cont : Continuous f) : IsLocalHomeomorph f :=
  isLocalHomeomorph_iff_isLocalHomeomorphOn_univ.mpr <|
    hgf.isLocalHomeomorphOn.of_comp_left hg.isLocalHomeomorphOn fun _ _ ↦ cont.continuousAt


theorem map_nhds_eq (hf : IsLocalHomeomorph f) (x : X) : (𝓝 x).map f = 𝓝 (f x) :=
  hf.isLocalHomeomorphOn.map_nhds_eq (Set.mem_univ x)


/-- A local homeomorphism is continuous. -/
protected theorem continuous (hf : IsLocalHomeomorph f) : Continuous f :=
  continuous_iff_continuousOn_univ.mpr hf.isLocalHomeomorphOn.continuousOn


/-- A local homeomorphism is an open map. -/
protected theorem isOpenMap (hf : IsLocalHomeomorph f) : IsOpenMap f :=
  IsOpenMap.of_nhds_le fun x ↦ ge_of_eq (hf.map_nhds_eq x)


/-- The composition of local homeomorphisms is a local homeomorphism. -/
protected theorem comp (hg : IsLocalHomeomorph g) (hf : IsLocalHomeomorph f) :
    IsLocalHomeomorph (g ∘ f) :=
  isLocalHomeomorph_iff_isLocalHomeomorphOn_univ.mpr
    (hg.isLocalHomeomorphOn.comp hf.isLocalHomeomorphOn (Set.univ.mapsTo_univ f))


/-- An injective local homeomorphism is an open embedding. -/
theorem isOpenEmbedding_of_injective (hf : IsLocalHomeomorph f) (hi : f.Injective) :
    IsOpenEmbedding f :=
  .of_continuous_injective_isOpenMap hf.continuous hi hf.isOpenMap


@[deprecated (since := "2024-10-18")]
alias openEmbedding_of_injective := isOpenEmbedding_of_injective


/-- A surjective embedding is a homeomorphism. -/
noncomputable def _root_.Topology.IsEmbedding.toHomeomorph_of_surjective (hf : IsEmbedding f)
    (hsurj : Function.Surjective f) : X ≃ₜ Y :=
  Homeomorph.homeomorphOfContinuousOpen (Equiv.ofBijective f ⟨hf.injective, hsurj⟩)
    hf.continuous (hf.isOpenEmbedding_of_surjective hsurj).isOpenMap


@[deprecated (since := "2024-10-26")]
alias _root_.Embedding.toHomeomeomorph_of_surjective := IsEmbedding.toHomeomorph_of_surjective


/-- A bijective local homeomorphism is a homeomorphism. -/
noncomputable def toHomeomorph_of_bijective (hf : IsLocalHomeomorph f) (hb : f.Bijective) :
    X ≃ₜ Y :=
  Homeomorph.homeomorphOfContinuousOpen (Equiv.ofBijective f hb) hf.continuous hf.isOpenMap


/-- Continuous local sections of a local homeomorphism are open embeddings. -/
theorem isOpenEmbedding_of_comp (hf : IsLocalHomeomorph g) (hgf : IsOpenEmbedding (g ∘ f))
    (cont : Continuous f) : IsOpenEmbedding f :=
  (hgf.isLocalHomeomorph.of_comp hf cont).isOpenEmbedding_of_injective hgf.injective.of_comp


@[deprecated (since := "2024-10-18")]
alias openEmbedding_of_comp := isOpenEmbedding_of_comp


open TopologicalSpace in
/-- Ranges of continuous local sections of a local homeomorphism
form a basis of the source space. -/
theorem isTopologicalBasis (hf : IsLocalHomeomorph f) : IsTopologicalBasis
    {U : Set X | ∃ V : Set Y, IsOpen V ∧ ∃ s : C(V,X), f ∘ s = (↑) ∧ Set.range s = U} := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : IsLocalHomeomorph f
    ⊢ TopologicalSpace.IsTopologicalBasis (setOf fun U => Exists fun V => And (IsO …
  -/
  refine isTopologicalBasis_of_isOpen_of_nhds ?_ fun x U hx hU ↦ ?_
    /-
      case refine_1
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      hf : IsLocalHomeomorph f
      ⊢ ∀ (u : Set X), Membership.mem (setOf fun U => Exists fun V => And (IsOpen V) …
    -/
  · rintro _ ⟨U, hU, s, hs, rfl⟩
    refine (isOpenEmbedding_of_comp hf (hs ▸ ⟨IsEmbedding.subtypeVal, ?_⟩)
      s.continuous).isOpen_range
    /-
      case refine_1.intro.intro.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      hf : IsLocalHomeomorph f
      U : Set Y
      hU : IsOpen U
      s : ContinuousMap (↑U) X
      hs : Eq (Function.comp f ⇑s) Subtype.val
      ⊢ IsOpen (Set.range Subtype.val)
    -/
    rwa [Subtype.range_val]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      hf : IsLocalHomeomorph f
      x : X
      U : Set X
      hx : Membership.mem U x
      hU : IsOpen U
      ⊢ Exists fun v => And (Membership.mem (setOf fun U => Exists fun V => And (IsO …
    -/
  · obtain ⟨f, hxf, rfl⟩ := hf x
    refine ⟨f.source ∩ U, ⟨f.target ∩ f.symm ⁻¹' U, f.symm.isOpen_inter_preimage hU,
      ⟨_, continuousOn_iff_continuous_restrict.mp (f.continuousOn_invFun.mono fun _ h ↦ h.1)⟩,
      ?_, (Set.range_restrict _ _).trans ?_⟩, ⟨hxf, hx⟩, fun _ h ↦ h.2⟩
      /-
        case refine_2.intro.intro.refine_1
        X : Type u_1
        Y : Type u_2
        inst✝¹ : TopologicalSpace X
        inst✝ : TopologicalSpace Y
        x : X
        U : Set X
        hx : Membership.mem U x
        hU : IsOpen U
        f : PartialHomeomorph X Y
        hxf : Membership.mem f.source x
        hf : IsLocalHomeomorph ↑f
        ⊢ Eq (Function.comp ↑f ⇑{ toFun := (Inter.inter f.target (Set.preimage (↑f.sym …
      -/
    · ext y; exact f.right_inv y.2.1
             /-
               🎉 no goals
             -/
      /-
        case refine_2.intro.intro.refine_2
        X : Type u_1
        Y : Type u_2
        inst✝¹ : TopologicalSpace X
        inst✝ : TopologicalSpace Y
        x : X
        U : Set X
        hx : Membership.mem U x
        hU : IsOpen U
        f : PartialHomeomorph X Y
        hxf : Membership.mem f.source x
        hf : IsLocalHomeomorph ↑f
        ⊢ Eq (Set.image f.toPartialEquiv.2 (Inter.inter f.target (Set.preimage (↑f.sym …
      -/
    · apply (f.symm_image_target_inter_eq _).trans
      rw [Set.preimage_inter, ← Set.inter_assoc, Set.inter_eq_self_of_subset_left
        f.source_preimage_target, f.source_inter_preimage_inv_preimage]


