/--
Let `X` be an object of a Galois category with fiber functor `F` and `Y` a sub-`Aut F`-set
of `F.obj X`, on which `Aut F` acts transitively (i.e. which is connected in the Galois category
of finite `Aut F`-sets). Then there exists a connected sub-object `Z` of `X` and an isomorphism
`Y ≅ F.obj X` as `Aut F`-sets such that the obvious triangle commutes.

For a version without the connectedness assumption, see `exists_lift_of_mono`.
-/
lemma exists_lift_of_mono_of_isConnected (X : C) (Y : Action FintypeCat.{u} (MonCat.of (Aut F)))
    (i : Y ⟶ (functorToAction F).obj X) [Mono i] [IsConnected Y] : ∃ (Z : C) (f : Z ⟶ X)
    (u : Y ≅ (functorToAction F).obj Z),
    IsConnected Z ∧ Mono f ∧ i = u.hom ≫ (functorToAction F).map f := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝³ : CategoryTheory.GaloisCategory C
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    Y : Action FintypeCat (MonCat.of (CategoryTheory.Aut F))
    i : Quiver.Hom Y ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X)
    inst✝¹ : CategoryTheory.Mono i
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected Y
    ⊢ Exists fun Z => Exists fun f => Exists fun u => And (CategoryTheory.PreGaloi …
  -/
  obtain ⟨y⟩ := nonempty_fiber_of_isConnected (forget₂ _ FintypeCat) Y
  /-
    case intro
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝³ : CategoryTheory.GaloisCategory C
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    Y : Action FintypeCat (MonCat.of (CategoryTheory.Aut F))
    i : Quiver.Hom Y ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X)
    inst✝¹ : CategoryTheory.Mono i
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected Y
    y : ↑((CategoryTheory.forget₂ (Action FintypeCat (MonCat.of (CategoryTheory.Au …
    ⊢ Exists fun Z => Exists fun f => Exists fun u => And (CategoryTheory.PreGaloi …
  -/
  obtain ⟨Z, f, z, hz, hc, hm⟩ := fiber_in_connected_component F X (i.hom y)
  /-
    case intro.intro.intro.intro.intro.intro
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝³ : CategoryTheory.GaloisCategory C
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    Y : Action FintypeCat (MonCat.of (CategoryTheory.Aut F))
    i : Quiver.Hom Y ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X)
    inst✝¹ : CategoryTheory.Mono i
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected Y
    y : ↑((CategoryTheory.forget₂ (Action FintypeCat (MonCat.of (CategoryTheory.Au …
    Z : C
    f : Quiver.Hom Z X
    z : ↑(F.obj Z)
    hz : Eq (F.map f z) (i.hom y)
    hc : CategoryTheory.PreGaloisCategory.IsConnected Z
    hm : CategoryTheory.Mono f
    ⊢ Exists fun Z => Exists fun f => Exists fun u => And (CategoryTheory.PreGaloi …
  -/
  have : IsConnected ((functorToAction F).obj Z) := PreservesIsConnected.preserves
  obtain ⟨u, hu⟩ := connected_component_unique
    (forget₂ (Action FintypeCat (MonCat.of (Aut F))) FintypeCat) (B := (functorToAction F).obj Z)
    y z i ((functorToAction F).map f) hz.symm
  /-
    case intro.intro.intro.intro.intro.intro.intro
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝³ : CategoryTheory.GaloisCategory C
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    Y : Action FintypeCat (MonCat.of (CategoryTheory.Aut F))
    i : Quiver.Hom Y ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X)
    inst✝¹ : CategoryTheory.Mono i
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected Y
    y : ↑((CategoryTheory.forget₂ (Action FintypeCat (MonCat.of (CategoryTheory.Au …
    Z : C
    f : Quiver.Hom Z X
    z : ↑(F.obj Z)
    hz : Eq (F.map f z) (i.hom y)
    hc : CategoryTheory.PreGaloisCategory.IsConnected Z
    hm : CategoryTheory.Mono f
    this : CategoryTheory.PreGaloisCategory.IsConnected ((CategoryTheory.PreGalois …
    u : CategoryTheory.Iso Y ((CategoryTheory.PreGaloisCategory.functorToAction F) …
    hu : Eq ((CategoryTheory.forget₂ (Action FintypeCat (MonCat.of (CategoryTheory …
    ⊢ Exists fun Z => Exists fun f => Exists fun u => And (CategoryTheory.PreGaloi …
  -/
  refine ⟨Z, f, u, hc, hm, ?_⟩
  apply evaluation_injective_of_isConnected
    (forget₂ (Action FintypeCat (MonCat.of (Aut F))) FintypeCat) Y ((functorToAction F).obj X) y
  /-
    case intro.intro.intro.intro.intro.intro.intro.a
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝³ : CategoryTheory.GaloisCategory C
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    Y : Action FintypeCat (MonCat.of (CategoryTheory.Aut F))
    i : Quiver.Hom Y ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X)
    inst✝¹ : CategoryTheory.Mono i
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected Y
    y : ↑((CategoryTheory.forget₂ (Action FintypeCat (MonCat.of (CategoryTheory.Au …
    Z : C
    f : Quiver.Hom Z X
    z : ↑(F.obj Z)
    hz : Eq (F.map f z) (i.hom y)
    hc : CategoryTheory.PreGaloisCategory.IsConnected Z
    hm : CategoryTheory.Mono f
    this : CategoryTheory.PreGaloisCategory.IsConnected ((CategoryTheory.PreGalois …
    u : CategoryTheory.Iso Y ((CategoryTheory.PreGaloisCategory.functorToAction F) …
    hu : Eq ((CategoryTheory.forget₂ (Action FintypeCat (MonCat.of (CategoryTheory …
    ⊢ Eq ((fun f => (CategoryTheory.forget₂ (Action FintypeCat (MonCat.of (Categor …
  -/
  suffices h : i.hom y = F.map f z by simpa [hu]
  /-
    case intro.intro.intro.intro.intro.intro.intro.a
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝³ : CategoryTheory.GaloisCategory C
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    Y : Action FintypeCat (MonCat.of (CategoryTheory.Aut F))
    i : Quiver.Hom Y ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X)
    inst✝¹ : CategoryTheory.Mono i
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected Y
    y : ↑((CategoryTheory.forget₂ (Action FintypeCat (MonCat.of (CategoryTheory.Au …
    Z : C
    f : Quiver.Hom Z X
    z : ↑(F.obj Z)
    hz : Eq (F.map f z) (i.hom y)
    hc : CategoryTheory.PreGaloisCategory.IsConnected Z
    hm : CategoryTheory.Mono f
    this : CategoryTheory.PreGaloisCategory.IsConnected ((CategoryTheory.PreGalois …
    u : CategoryTheory.Iso Y ((CategoryTheory.PreGaloisCategory.functorToAction F) …
    hu : Eq ((CategoryTheory.forget₂ (Action FintypeCat (MonCat.of (CategoryTheory …
    ⊢ Eq (i.hom y) (F.map f z)
  -/
  exact hz.symm
  /-
    🎉 no goals
  -/


/--
Let `X` be an object of a Galois category with fiber functor `F` and `Y` a sub-`Aut F`-set
of `F.obj X`. Then there exists a sub-object `Z` of `X` and an isomorphism
`Y ≅ F.obj X` as `Aut F`-sets such that the obvious triangle commutes.
-/
lemma exists_lift_of_mono (X : C) (Y : Action FintypeCat.{u} (MonCat.of (Aut F)))
    (i : Y ⟶ (functorToAction F).obj X) [Mono i] : ∃ (Z : C) (f : Z ⟶ X)
    (u : Y ≅ (functorToAction F).obj Z), Mono f ∧ u.hom ≫ (functorToAction F).map f = i := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.GaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    Y : Action FintypeCat (MonCat.of (CategoryTheory.Aut F))
    i : Quiver.Hom Y ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X)
    inst✝ : CategoryTheory.Mono i
    ⊢ Exists fun Z => Exists fun f => Exists fun u => And (CategoryTheory.Mono f)  …
  -/
  obtain ⟨ι, hf, f, t, hc⟩ := has_decomp_connected_components' Y
  /-
    case intro.intro.intro.intro
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.GaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    Y : Action FintypeCat (MonCat.of (CategoryTheory.Aut F))
    i : Quiver.Hom Y ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X)
    inst✝ : CategoryTheory.Mono i
    ι : Type
    hf : Finite ι
    f : ι → Action FintypeCat (MonCat.of (CategoryTheory.Aut F))
    t : CategoryTheory.Iso (CategoryTheory.Limits.sigmaObj f) Y
    hc : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsConnected (f i)
    ⊢ Exists fun Z => Exists fun f => Exists fun u => And (CategoryTheory.Mono f)  …
  -/
  let i' (j : ι) : f j ⟶ (functorToAction F).obj X := Sigma.ι f j ≫ t.hom ≫ i
  have (j : ι) : Mono (i' j) :=
    have : Mono (Sigma.ι f j) := MonoCoprod.mono_ι f j
    have : Mono (t.hom ≫ i) := mono_comp _ _
    mono_comp _ _
  /-
    case intro.intro.intro.intro
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.GaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    Y : Action FintypeCat (MonCat.of (CategoryTheory.Aut F))
    i : Quiver.Hom Y ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X)
    inst✝ : CategoryTheory.Mono i
    ι : Type
    hf : Finite ι
    f : ι → Action FintypeCat (MonCat.of (CategoryTheory.Aut F))
    t : CategoryTheory.Iso (CategoryTheory.Limits.sigmaObj f) Y
    hc : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsConnected (f i)
    i' : (j : ι) → Quiver.Hom (f j) ((CategoryTheory.PreGaloisCategory.functorToAc …
    this : ∀ (j : ι), CategoryTheory.Mono (i' j)
    ⊢ Exists fun Z => Exists fun f => Exists fun u => And (CategoryTheory.Mono f)  …
  -/
  choose gZ gf gu _ _ h using fun i ↦ exists_lift_of_mono_of_isConnected F X (f i) (i' i)
  let is2 : (functorToAction F).obj (∐ gZ) ≅ ∐ fun i => (functorToAction F).obj (gZ i) :=
    PreservesCoproduct.iso (functorToAction F) gZ
  /-
    case intro.intro.intro.intro
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.GaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    Y : Action FintypeCat (MonCat.of (CategoryTheory.Aut F))
    i : Quiver.Hom Y ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X)
    inst✝ : CategoryTheory.Mono i
    ι : Type
    hf : Finite ι
    f : ι → Action FintypeCat (MonCat.of (CategoryTheory.Aut F))
    t : CategoryTheory.Iso (CategoryTheory.Limits.sigmaObj f) Y
    hc : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsConnected (f i)
    i' : (j : ι) → Quiver.Hom (f j) ((CategoryTheory.PreGaloisCategory.functorToAc …
    this : ∀ (j : ι), CategoryTheory.Mono (i' j)
    gZ : ι → C
    gf : (i : ι) → Quiver.Hom (gZ i) X
    gu : (i : ι) → CategoryTheory.Iso (f i) ((CategoryTheory.PreGaloisCategory.fun …
    h✝¹ : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsConnected (gZ i)
    h✝ : ∀ (i : ι), CategoryTheory.Mono (gf i)
    h : ∀ (i : ι), Eq (i' i) (CategoryTheory.CategoryStruct.comp (gu i).hom ((Cate …
    is2 : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F) …
    ⊢ Exists fun Z => Exists fun f => Exists fun u => And (CategoryTheory.Mono f)  …
  -/
  let u' : ∐ f ≅ ∐ fun i => (functorToAction F).obj (gZ i) := Sigma.mapIso gu
  have heq : (functorToAction F).map (Sigma.desc gf) = (t.symm ≪≫ u' ≪≫ is2.symm).inv ≫ i := by
    simp only [Iso.trans_inv, Iso.symm_inv, Category.assoc]
    rw [← Iso.inv_comp_eq]
    refine Sigma.hom_ext _ _ (fun j ↦ ?_)
    suffices (functorToAction F).map (gf j) = (gu j).inv ≫ i' j by
      simpa [is2, u']
    simp only [h, Iso.inv_hom_id_assoc]
  /-
    case intro.intro.intro.intro
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.GaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    Y : Action FintypeCat (MonCat.of (CategoryTheory.Aut F))
    i : Quiver.Hom Y ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X)
    inst✝ : CategoryTheory.Mono i
    ι : Type
    hf : Finite ι
    f : ι → Action FintypeCat (MonCat.of (CategoryTheory.Aut F))
    t : CategoryTheory.Iso (CategoryTheory.Limits.sigmaObj f) Y
    hc : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsConnected (f i)
    i' : (j : ι) → Quiver.Hom (f j) ((CategoryTheory.PreGaloisCategory.functorToAc …
    this : ∀ (j : ι), CategoryTheory.Mono (i' j)
    gZ : ι → C
    gf : (i : ι) → Quiver.Hom (gZ i) X
    gu : (i : ι) → CategoryTheory.Iso (f i) ((CategoryTheory.PreGaloisCategory.fun …
    h✝¹ : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsConnected (gZ i)
    h✝ : ∀ (i : ι), CategoryTheory.Mono (gf i)
    h : ∀ (i : ι), Eq (i' i) (CategoryTheory.CategoryStruct.comp (gu i).hom ((Cate …
    is2 : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F) …
    u' : CategoryTheory.Iso (CategoryTheory.Limits.sigmaObj f) (CategoryTheory.Lim …
    heq : Eq ((CategoryTheory.PreGaloisCategory.functorToAction F).map (CategoryTh …
    ⊢ Exists fun Z => Exists fun f => Exists fun u => And (CategoryTheory.Mono f)  …
  -/
  refine ⟨∐ gZ, Sigma.desc gf, t.symm ≪≫ u' ≪≫ is2.symm, ?_, by simp [heq]⟩
    /-
      case intro.intro.intro.intro
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝² : CategoryTheory.GaloisCategory C
      inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X : C
      Y : Action FintypeCat (MonCat.of (CategoryTheory.Aut F))
      i : Quiver.Hom Y ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X)
      inst✝ : CategoryTheory.Mono i
      ι : Type
      hf : Finite ι
      f : ι → Action FintypeCat (MonCat.of (CategoryTheory.Aut F))
      t : CategoryTheory.Iso (CategoryTheory.Limits.sigmaObj f) Y
      hc : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsConnected (f i)
      i' : (j : ι) → Quiver.Hom (f j) ((CategoryTheory.PreGaloisCategory.functorToAc …
      this : ∀ (j : ι), CategoryTheory.Mono (i' j)
      gZ : ι → C
      gf : (i : ι) → Quiver.Hom (gZ i) X
      gu : (i : ι) → CategoryTheory.Iso (f i) ((CategoryTheory.PreGaloisCategory.fun …
      h✝¹ : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsConnected (gZ i)
      h✝ : ∀ (i : ι), CategoryTheory.Mono (gf i)
      h : ∀ (i : ι), Eq (i' i) (CategoryTheory.CategoryStruct.comp (gu i).hom ((Cate …
      is2 : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F) …
      u' : CategoryTheory.Iso (CategoryTheory.Limits.sigmaObj f) (CategoryTheory.Lim …
      heq : Eq ((CategoryTheory.PreGaloisCategory.functorToAction F).map (CategoryTh …
      ⊢ CategoryTheory.Mono (CategoryTheory.Limits.Sigma.desc gf)
    -/
  · exact mono_of_mono_map (functorToAction F) (heq ▸ mono_comp _ _)
    /-
      🎉 no goals
    -/


/-- The by a fiber functor `F : C ⥤ FintypeCat` induced functor `functorToAction F` to
finite `Aut F`-sets is full. -/
instance functorToAction_full : Functor.Full (functorToAction F) where
  map_surjective {X Y} f := by
    let u : (functorToAction F).obj X ⟶ (functorToAction F).obj X ⨯ (functorToAction F).obj Y :=
      prod.lift (𝟙 _) f
    let i : (functorToAction F).obj X ⟶ (functorToAction F).obj (X ⨯ Y) :=
      u ≫ (PreservesLimitPair.iso (functorToAction F) X Y).inv
    have : Mono i := by
      have : Mono (u ≫ prod.fst) := prod.lift_fst (𝟙 _) f ▸ inferInstance
      have : Mono u := mono_of_mono u prod.fst
      apply mono_comp u _
    obtain ⟨Z, g, v, _, hvgi⟩ := exists_lift_of_mono F (Limits.prod X Y)
      ((functorToAction F).obj X) i
    /-
      case intro.intro.intro.intro
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.GaloisCategory C
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X Y : C
      f : Quiver.Hom ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X) (( …
      u : Quiver.Hom ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X) (C …
      i : Quiver.Hom ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X) (( …
      this : CategoryTheory.Mono i
      Z : C
      g : Quiver.Hom Z (CategoryTheory.Limits.prod X Y)
      v : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      left✝ : CategoryTheory.Mono g
      hvgi : Eq (CategoryTheory.CategoryStruct.comp v.hom ((CategoryTheory.PreGalois …
      ⊢ Exists fun a => Eq ((CategoryTheory.PreGaloisCategory.functorToAction F).map …
    -/
    let ψ : Z ⟶ X := g ≫ prod.fst
    /-
      case intro.intro.intro.intro
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.GaloisCategory C
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X Y : C
      f : Quiver.Hom ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X) (( …
      u : Quiver.Hom ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X) (C …
      i : Quiver.Hom ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X) (( …
      this : CategoryTheory.Mono i
      Z : C
      g : Quiver.Hom Z (CategoryTheory.Limits.prod X Y)
      v : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      left✝ : CategoryTheory.Mono g
      hvgi : Eq (CategoryTheory.CategoryStruct.comp v.hom ((CategoryTheory.PreGalois …
      ψ : Quiver.Hom Z X := CategoryTheory.CategoryStruct.comp g CategoryTheory.Limi …
      ⊢ Exists fun a => Eq ((CategoryTheory.PreGaloisCategory.functorToAction F).map …
    -/
    have hgvi : (functorToAction F).map g = v.inv ≫ i := by simp [← hvgi]
    have : IsIso ((functorToAction F).map ψ) := by
      simp only [map_comp, hgvi, Category.assoc, ψ]
      have : IsIso (i ≫ (functorToAction F).map prod.fst) := by
        suffices h : IsIso (𝟙 ((functorToAction F).obj X)) by simpa [i, u]
        infer_instance
      apply IsIso.comp_isIso
    /-
      case intro.intro.intro.intro
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.GaloisCategory C
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X Y : C
      f : Quiver.Hom ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X) (( …
      u : Quiver.Hom ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X) (C …
      i : Quiver.Hom ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X) (( …
      this✝ : CategoryTheory.Mono i
      Z : C
      g : Quiver.Hom Z (CategoryTheory.Limits.prod X Y)
      v : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      left✝ : CategoryTheory.Mono g
      hvgi : Eq (CategoryTheory.CategoryStruct.comp v.hom ((CategoryTheory.PreGalois …
      ψ : Quiver.Hom Z X := CategoryTheory.CategoryStruct.comp g CategoryTheory.Limi …
      hgvi : Eq ((CategoryTheory.PreGaloisCategory.functorToAction F).map g) (Catego …
      this : CategoryTheory.IsIso ((CategoryTheory.PreGaloisCategory.functorToAction …
      ⊢ Exists fun a => Eq ((CategoryTheory.PreGaloisCategory.functorToAction F).map …
    -/
    have : IsIso ψ := isIso_of_reflects_iso ψ (functorToAction F)
    /-
      case intro.intro.intro.intro
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.GaloisCategory C
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X Y : C
      f : Quiver.Hom ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X) (( …
      u : Quiver.Hom ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X) (C …
      i : Quiver.Hom ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X) (( …
      this✝¹ : CategoryTheory.Mono i
      Z : C
      g : Quiver.Hom Z (CategoryTheory.Limits.prod X Y)
      v : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      left✝ : CategoryTheory.Mono g
      hvgi : Eq (CategoryTheory.CategoryStruct.comp v.hom ((CategoryTheory.PreGalois …
      ψ : Quiver.Hom Z X := CategoryTheory.CategoryStruct.comp g CategoryTheory.Limi …
      hgvi : Eq ((CategoryTheory.PreGaloisCategory.functorToAction F).map g) (Catego …
      this✝ : CategoryTheory.IsIso ((CategoryTheory.PreGaloisCategory.functorToActio …
      this : CategoryTheory.IsIso ψ
      ⊢ Exists fun a => Eq ((CategoryTheory.PreGaloisCategory.functorToAction F).map …
    -/
    use inv ψ ≫ g ≫ prod.snd
    /-
      case h
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.GaloisCategory C
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X Y : C
      f : Quiver.Hom ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X) (( …
      u : Quiver.Hom ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X) (C …
      i : Quiver.Hom ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X) (( …
      this✝¹ : CategoryTheory.Mono i
      Z : C
      g : Quiver.Hom Z (CategoryTheory.Limits.prod X Y)
      v : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      left✝ : CategoryTheory.Mono g
      hvgi : Eq (CategoryTheory.CategoryStruct.comp v.hom ((CategoryTheory.PreGalois …
      ψ : Quiver.Hom Z X := CategoryTheory.CategoryStruct.comp g CategoryTheory.Limi …
      hgvi : Eq ((CategoryTheory.PreGaloisCategory.functorToAction F).map g) (Catego …
      this✝ : CategoryTheory.IsIso ((CategoryTheory.PreGaloisCategory.functorToActio …
      this : CategoryTheory.IsIso ψ
      ⊢ Eq ((CategoryTheory.PreGaloisCategory.functorToAction F).map (CategoryTheory …
    -/
    rw [← cancel_epi ((functorToAction F).map ψ)]
    /-
      case h
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.GaloisCategory C
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X Y : C
      f : Quiver.Hom ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X) (( …
      u : Quiver.Hom ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X) (C …
      i : Quiver.Hom ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X) (( …
      this✝¹ : CategoryTheory.Mono i
      Z : C
      g : Quiver.Hom Z (CategoryTheory.Limits.prod X Y)
      v : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      left✝ : CategoryTheory.Mono g
      hvgi : Eq (CategoryTheory.CategoryStruct.comp v.hom ((CategoryTheory.PreGalois …
      ψ : Quiver.Hom Z X := CategoryTheory.CategoryStruct.comp g CategoryTheory.Limi …
      hgvi : Eq ((CategoryTheory.PreGaloisCategory.functorToAction F).map g) (Catego …
      this✝ : CategoryTheory.IsIso ((CategoryTheory.PreGaloisCategory.functorToActio …
      this : CategoryTheory.IsIso ψ
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.PreGaloisCategory.fu …
    -/
    ext (z : F.obj Z)
    /-
      case h.h.h
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.GaloisCategory C
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X Y : C
      f : Quiver.Hom ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X) (( …
      u : Quiver.Hom ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X) (C …
      i : Quiver.Hom ((CategoryTheory.PreGaloisCategory.functorToAction F).obj X) (( …
      this✝¹ : CategoryTheory.Mono i
      Z : C
      g : Quiver.Hom Z (CategoryTheory.Limits.prod X Y)
      v : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      left✝ : CategoryTheory.Mono g
      hvgi : Eq (CategoryTheory.CategoryStruct.comp v.hom ((CategoryTheory.PreGalois …
      ψ : Quiver.Hom Z X := CategoryTheory.CategoryStruct.comp g CategoryTheory.Limi …
      hgvi : Eq ((CategoryTheory.PreGaloisCategory.functorToAction F).map g) (Catego …
      this✝ : CategoryTheory.IsIso ((CategoryTheory.PreGaloisCategory.functorToActio …
      this : CategoryTheory.IsIso ψ
      z : ↑(F.obj Z)
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.PreGaloisCategory.f …
    -/
    simp [-FintypeCat.comp_apply, -Action.comp_hom, i, u, ψ, hgvi]
    /-
      🎉 no goals
    -/


