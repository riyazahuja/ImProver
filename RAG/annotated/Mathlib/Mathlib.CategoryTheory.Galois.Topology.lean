/-- For a functor `F : C ⥤ FintypeCat`, the canonical embedding of `Aut F` into
the product over `Aut (F.obj X)` for all objects `X`. -/
def autEmbedding : Aut F →* ∀ X, Aut (F.obj X) :=
  MonoidHom.mk' (fun σ X ↦ σ.app X) (fun _ _ ↦ rfl)


@[simp]
lemma autEmbedding_apply (σ : Aut F) (X : C) : autEmbedding F σ X = σ.app X :=
  rfl


lemma autEmbedding_injective : Function.Injective (autEmbedding F) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    ⊢ Function.Injective ⇑(CategoryTheory.PreGaloisCategory.autEmbedding F)
  -/
  intro σ τ h
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    σ τ : CategoryTheory.Aut F
    h : Eq ((CategoryTheory.PreGaloisCategory.autEmbedding F) σ) ((CategoryTheory. …
    ⊢ Eq σ τ
  -/
  ext X x
  /-
    case h.w.h.h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    σ τ : CategoryTheory.Aut F
    h : Eq ((CategoryTheory.PreGaloisCategory.autEmbedding F) σ) ((CategoryTheory. …
    X : C
    x : ↑(F.obj X)
    ⊢ Eq (σ.hom.app X x) (τ.hom.app X x)
  -/
  have : σ.app X = τ.app X := congr_fun h X
  /-
    case h.w.h.h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    σ τ : CategoryTheory.Aut F
    h : Eq ((CategoryTheory.PreGaloisCategory.autEmbedding F) σ) ((CategoryTheory. …
    X : C
    x : ↑(F.obj X)
    this : Eq (CategoryTheory.Iso.app σ X) (CategoryTheory.Iso.app τ X)
    ⊢ Eq (σ.hom.app X x) (τ.hom.app X x)
  -/
  rw [← Iso.app_hom, ← Iso.app_hom, this]
  /-
    🎉 no goals
  -/


/-- We put the discrete topology on `F.obj X`. -/
scoped instance (X : C) : TopologicalSpace (F.obj X) := ⊥


@[scoped instance]
lemma obj_discreteTopology (X : C) : DiscreteTopology (F.obj X) := ⟨rfl⟩


/-- We put the discrete topology on `Aut (F.obj X)`. -/
scoped instance (X : C) : TopologicalSpace (Aut (F.obj X)) := ⊥


@[scoped instance]
lemma aut_discreteTopology (X : C) : DiscreteTopology (Aut (F.obj X)) := ⟨rfl⟩


/-- `Aut F` is equipped with the by the embedding into `∀ X, Aut (F.obj X)` induced embedding. -/
instance : TopologicalSpace (Aut F) :=
  TopologicalSpace.induced (autEmbedding F) inferInstance


/-- The image of `Aut F` in `∀ X, Aut (F.obj X)` are precisely the compatible families of
automorphisms. -/
lemma autEmbedding_range :
    Set.range (autEmbedding F) =
      ⋂ (f : Arrow C), { a | F.map f.hom ≫ (a f.right).hom = (a f.left).hom ≫ F.map f.hom } := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    ⊢ Eq (Set.range ⇑(CategoryTheory.PreGaloisCategory.autEmbedding F)) (Set.iInte …
  -/
  ext a
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    a : (X : C) → CategoryTheory.Aut (F.obj X)
    ⊢ Iff (Membership.mem (Set.range ⇑(CategoryTheory.PreGaloisCategory.autEmbeddi …
  -/
  simp only [Set.mem_range, id_obj, Set.mem_iInter, Set.mem_setOf_eq]
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    a : (X : C) → CategoryTheory.Aut (F.obj X)
    ⊢ Iff (Exists fun y => Eq ((CategoryTheory.PreGaloisCategory.autEmbedding F) y …
  -/
  refine ⟨fun ⟨σ, h⟩ i ↦ h.symm ▸ σ.hom.naturality i.hom, fun h ↦ ?_⟩
    /-
      case h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      a : (X : C) → CategoryTheory.Aut (F.obj X)
      h : ∀ (i : CategoryTheory.Arrow C), Eq (CategoryTheory.CategoryStruct.comp (F. …
      ⊢ Exists fun y => Eq ((CategoryTheory.PreGaloisCategory.autEmbedding F) y) a
    -/
  · use NatIso.ofComponents a (fun {X Y} f ↦ h ⟨X, Y, f⟩)
    /-
      case h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      a : (X : C) → CategoryTheory.Aut (F.obj X)
      h : ∀ (i : CategoryTheory.Arrow C), Eq (CategoryTheory.CategoryStruct.comp (F. …
      ⊢ Eq ((CategoryTheory.PreGaloisCategory.autEmbedding F) (CategoryTheory.NatIso …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The image of `Aut F` in `∀ X, Aut (F.obj X)` is closed. -/
lemma autEmbedding_range_isClosed : IsClosed (Set.range (autEmbedding F)) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    ⊢ IsClosed (Set.range ⇑(CategoryTheory.PreGaloisCategory.autEmbedding F))
  -/
  rw [autEmbedding_range]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    ⊢ IsClosed (Set.iInter fun f => setOf fun a => Eq (CategoryTheory.CategoryStru …
  -/
  refine isClosed_iInter (fun f ↦ isClosed_eq (X := F.obj f.left → F.obj f.right) ?_ ?_)
    /-
      case refine_1
      C : Type u₁
      inst✝ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      f : CategoryTheory.Arrow C
      ⊢ Continuous fun a => CategoryTheory.CategoryStruct.comp (F.map f.hom) (a f.ri …
    -/
  · fun_prop
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u₁
      inst✝ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      f : CategoryTheory.Arrow C
      ⊢ Continuous fun a => CategoryTheory.CategoryStruct.comp (a f.left).hom (F.map …
    -/
  · fun_prop
    /-
      🎉 no goals
    -/


lemma autEmbedding_isClosedEmbedding : IsClosedEmbedding (autEmbedding F) where
  eq_induced := rfl
  injective := autEmbedding_injective F
  isClosed_range := autEmbedding_range_isClosed F


@[deprecated (since := "2024-10-20")]
alias autEmbedding_closedEmbedding := autEmbedding_isClosedEmbedding


instance : CompactSpace (Aut F) := (autEmbedding_isClosedEmbedding F).compactSpace


instance : T2Space (Aut F) :=
  T2Space.of_injective_continuous (autEmbedding_injective F) continuous_induced_dom


instance : TotallyDisconnectedSpace (Aut F) :=
  (autEmbedding_isClosedEmbedding F).isEmbedding.isTotallyDisconnected_range.mp
    (isTotallyDisconnected_of_totallyDisconnectedSpace _)


instance : ContinuousMul (Aut F) :=
  (autEmbedding_isClosedEmbedding F).isInducing.continuousMul (autEmbedding F)


instance : ContinuousInv (Aut F) :=
  (autEmbedding_isClosedEmbedding F).isInducing.continuousInv fun _ ↦ rfl


instance : TopologicalGroup (Aut F) := ⟨⟩


instance (X : C) : SMul (Aut (F.obj X)) (F.obj X) := ⟨fun σ a => σ.hom a⟩


instance (X : C) : ContinuousSMul (Aut (F.obj X)) (F.obj X) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    X : C
    ⊢ ContinuousSMul (CategoryTheory.Aut (F.obj X)) ↑(F.obj X)
  -/
  constructor
  /-
    case continuous_smul
    C : Type u₁
    inst✝ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    X : C
    ⊢ Continuous fun p => HSMul.hSMul p.1 p.2
  -/
  fun_prop
  /-
    🎉 no goals
  -/


instance continuousSMul_aut_fiber (X : C) : ContinuousSMul (Aut F) (F.obj X) where
  continuous_smul := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      X : C
      ⊢ Continuous fun p => HSMul.hSMul p.1 p.2
    -/
    let g : Aut (F.obj X) × F.obj X → F.obj X := fun ⟨σ, x⟩ ↦ σ.hom x
    let h (q : Aut F × F.obj X) : Aut (F.obj X) × F.obj X :=
      ⟨((fun p ↦ p X) ∘ autEmbedding F) q.1, q.2⟩
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      X : C
      g : Prod (CategoryTheory.Aut (F.obj X)) ↑(F.obj X) → ↑(F.obj X) := fun x => Ca …
      h : Prod (CategoryTheory.Aut F) ↑(F.obj X) → Prod (CategoryTheory.Aut (F.obj X …
      ⊢ Continuous fun p => HSMul.hSMul p.1 p.2
    -/
    show Continuous (g ∘ h)
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      X : C
      g : Prod (CategoryTheory.Aut (F.obj X)) ↑(F.obj X) → ↑(F.obj X) := fun x => Ca …
      h : Prod (CategoryTheory.Aut F) ↑(F.obj X) → Prod (CategoryTheory.Aut (F.obj X …
      ⊢ Continuous (Function.comp g h)
    -/
    fun_prop
    /-
      🎉 no goals
    -/


/--
If `H` is an open subset of `Aut F` such that `1 ∈ H`, there exists a finite
set `I` of connected objects of `C` such that every `σ : Aut F` that induces the identity
on `F.obj X` for all `X ∈ I` is contained in `H`. In other words: The kernel
of the evaluation map `Aut F →* ∏ X : I ↦ Aut (F.obj X)` is contained in `H`.
-/
lemma exists_set_ker_evaluation_subset_of_isOpen
    {H : Set (Aut F)} (h1 : 1 ∈ H) (h : IsOpen H) :
    ∃ (I : Set C) (_ : Fintype I), (∀ X ∈ I, IsConnected X) ∧
      (∀ σ : Aut F, (∀ X : I, σ.hom.app X = 𝟙 (F.obj X)) → σ ∈ H) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.GaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    H : Set (CategoryTheory.Aut F)
    h1 : Membership.mem H 1
    h : IsOpen H
    ⊢ Exists fun I => Exists fun x => And (∀ (X : C), Membership.mem I X → Categor …
  -/
  obtain ⟨U, hUopen, rfl⟩ := isOpen_induced_iff.mp h
  /-
    case intro.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.GaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    U : Set ((X : C) → CategoryTheory.Aut (F.obj X))
    hUopen : IsOpen U
    h1 : Membership.mem (Set.preimage (⇑(CategoryTheory.PreGaloisCategory.autEmbed …
    h : IsOpen (Set.preimage (⇑(CategoryTheory.PreGaloisCategory.autEmbedding F)) U)
    ⊢ Exists fun I => Exists fun x => And (∀ (X : C), Membership.mem I X → Categor …
  -/
  obtain ⟨I, u, ho, ha⟩ := isOpen_pi_iff.mp hUopen 1 h1
  /-
    case intro.intro.intro.intro.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.GaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    U : Set ((X : C) → CategoryTheory.Aut (F.obj X))
    hUopen : IsOpen U
    h1 : Membership.mem (Set.preimage (⇑(CategoryTheory.PreGaloisCategory.autEmbed …
    h : IsOpen (Set.preimage (⇑(CategoryTheory.PreGaloisCategory.autEmbedding F)) U)
    I : Finset C
    u : (a : C) → Set (CategoryTheory.Aut (F.obj a))
    ho : ∀ (a : C), Membership.mem I a → And (IsOpen (u a)) (Membership.mem (u a)  …
    ha : HasSubset.Subset ((↑I).pi u) U
    ⊢ Exists fun I => Exists fun x => And (∀ (X : C), Membership.mem I X → Categor …
  -/
  choose fι ff fc h4 h5 h6 using (fun X : I => has_decomp_connected_components X.val)
  /-
    case intro.intro.intro.intro.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.GaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    U : Set ((X : C) → CategoryTheory.Aut (F.obj X))
    hUopen : IsOpen U
    h1 : Membership.mem (Set.preimage (⇑(CategoryTheory.PreGaloisCategory.autEmbed …
    h : IsOpen (Set.preimage (⇑(CategoryTheory.PreGaloisCategory.autEmbedding F)) U)
    I : Finset C
    u : (a : C) → Set (CategoryTheory.Aut (F.obj a))
    ho : ∀ (a : C), Membership.mem I a → And (IsOpen (u a)) (Membership.mem (u a)  …
    ha : HasSubset.Subset ((↑I).pi u) U
    fι : (Subtype fun x => Membership.mem I x) → Type
    ff : (X : Subtype fun x => Membership.mem I x) → fι X → C
    fc : (X : Subtype fun x => Membership.mem I x) → (i : fι X) → Quiver.Hom (ff X …
    h4 : (X : Subtype fun x => Membership.mem I x) → CategoryTheory.Limits.IsColim …
    h5 : ∀ (X : Subtype fun x => Membership.mem I x) (i : fι X), CategoryTheory.Pr …
    h6 : ∀ (X : Subtype fun x => Membership.mem I x), Finite (fι X)
    ⊢ Exists fun I => Exists fun x => And (∀ (X : C), Membership.mem I X → Categor …
  -/
  refine ⟨⋃ X, Set.range (ff X), Fintype.ofFinite _, ?_, ?_⟩
    /-
      case intro.intro.intro.intro.intro.refine_1
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.GaloisCategory C
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      U : Set ((X : C) → CategoryTheory.Aut (F.obj X))
      hUopen : IsOpen U
      h1 : Membership.mem (Set.preimage (⇑(CategoryTheory.PreGaloisCategory.autEmbed …
      h : IsOpen (Set.preimage (⇑(CategoryTheory.PreGaloisCategory.autEmbedding F)) U)
      I : Finset C
      u : (a : C) → Set (CategoryTheory.Aut (F.obj a))
      ho : ∀ (a : C), Membership.mem I a → And (IsOpen (u a)) (Membership.mem (u a)  …
      ha : HasSubset.Subset ((↑I).pi u) U
      fι : (Subtype fun x => Membership.mem I x) → Type
      ff : (X : Subtype fun x => Membership.mem I x) → fι X → C
      fc : (X : Subtype fun x => Membership.mem I x) → (i : fι X) → Quiver.Hom (ff X …
      h4 : (X : Subtype fun x => Membership.mem I x) → CategoryTheory.Limits.IsColim …
      h5 : ∀ (X : Subtype fun x => Membership.mem I x) (i : fι X), CategoryTheory.Pr …
      h6 : ∀ (X : Subtype fun x => Membership.mem I x), Finite (fι X)
      ⊢ ∀ (X : C), Membership.mem (Set.iUnion fun X => Set.range (ff X)) X → Categor …
    -/
  · rintro X ⟨A, ⟨Y, rfl⟩, hA2⟩
    /-
      case intro.intro.intro.intro.intro.refine_1.intro.intro.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.GaloisCategory C
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      U : Set ((X : C) → CategoryTheory.Aut (F.obj X))
      hUopen : IsOpen U
      h1 : Membership.mem (Set.preimage (⇑(CategoryTheory.PreGaloisCategory.autEmbed …
      h : IsOpen (Set.preimage (⇑(CategoryTheory.PreGaloisCategory.autEmbedding F)) U)
      I : Finset C
      u : (a : C) → Set (CategoryTheory.Aut (F.obj a))
      ho : ∀ (a : C), Membership.mem I a → And (IsOpen (u a)) (Membership.mem (u a)  …
      ha : HasSubset.Subset ((↑I).pi u) U
      fι : (Subtype fun x => Membership.mem I x) → Type
      ff : (X : Subtype fun x => Membership.mem I x) → fι X → C
      fc : (X : Subtype fun x => Membership.mem I x) → (i : fι X) → Quiver.Hom (ff X …
      h4 : (X : Subtype fun x => Membership.mem I x) → CategoryTheory.Limits.IsColim …
      h5 : ∀ (X : Subtype fun x => Membership.mem I x) (i : fι X), CategoryTheory.Pr …
      h6 : ∀ (X : Subtype fun x => Membership.mem I x), Finite (fι X)
      X : C
      Y : Subtype fun x => Membership.mem I x
      hA2 : Membership.mem ((fun X => Set.range (ff X)) Y) X
      ⊢ CategoryTheory.PreGaloisCategory.IsConnected X
    -/
    obtain ⟨i, rfl⟩ := hA2
    /-
      case intro.intro.intro.intro.intro.refine_1.intro.intro.intro.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.GaloisCategory C
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      U : Set ((X : C) → CategoryTheory.Aut (F.obj X))
      hUopen : IsOpen U
      h1 : Membership.mem (Set.preimage (⇑(CategoryTheory.PreGaloisCategory.autEmbed …
      h : IsOpen (Set.preimage (⇑(CategoryTheory.PreGaloisCategory.autEmbedding F)) U)
      I : Finset C
      u : (a : C) → Set (CategoryTheory.Aut (F.obj a))
      ho : ∀ (a : C), Membership.mem I a → And (IsOpen (u a)) (Membership.mem (u a)  …
      ha : HasSubset.Subset ((↑I).pi u) U
      fι : (Subtype fun x => Membership.mem I x) → Type
      ff : (X : Subtype fun x => Membership.mem I x) → fι X → C
      fc : (X : Subtype fun x => Membership.mem I x) → (i : fι X) → Quiver.Hom (ff X …
      h4 : (X : Subtype fun x => Membership.mem I x) → CategoryTheory.Limits.IsColim …
      h5 : ∀ (X : Subtype fun x => Membership.mem I x) (i : fι X), CategoryTheory.Pr …
      h6 : ∀ (X : Subtype fun x => Membership.mem I x), Finite (fι X)
      Y : Subtype fun x => Membership.mem I x
      i : fι Y
      ⊢ CategoryTheory.PreGaloisCategory.IsConnected (ff Y i)
    -/
    exact h5 Y i
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.refine_2
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.GaloisCategory C
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      U : Set ((X : C) → CategoryTheory.Aut (F.obj X))
      hUopen : IsOpen U
      h1 : Membership.mem (Set.preimage (⇑(CategoryTheory.PreGaloisCategory.autEmbed …
      h : IsOpen (Set.preimage (⇑(CategoryTheory.PreGaloisCategory.autEmbedding F)) U)
      I : Finset C
      u : (a : C) → Set (CategoryTheory.Aut (F.obj a))
      ho : ∀ (a : C), Membership.mem I a → And (IsOpen (u a)) (Membership.mem (u a)  …
      ha : HasSubset.Subset ((↑I).pi u) U
      fι : (Subtype fun x => Membership.mem I x) → Type
      ff : (X : Subtype fun x => Membership.mem I x) → fι X → C
      fc : (X : Subtype fun x => Membership.mem I x) → (i : fι X) → Quiver.Hom (ff X …
      h4 : (X : Subtype fun x => Membership.mem I x) → CategoryTheory.Limits.IsColim …
      h5 : ∀ (X : Subtype fun x => Membership.mem I x) (i : fι X), CategoryTheory.Pr …
      h6 : ∀ (X : Subtype fun x => Membership.mem I x), Finite (fι X)
      ⊢ ∀ (σ : CategoryTheory.Aut F), (∀ (X : ↑(Set.iUnion fun X => Set.range (ff X) …
    -/
  · refine fun σ h ↦ ha (fun X XinI ↦ ?_)
    suffices h : autEmbedding F σ X = 1 by
      rw [h]
      exact (ho X XinI).right
    have h : σ.hom.app X = 𝟙 (F.obj X) := by
      have : Fintype (fι ⟨X, XinI⟩) := Fintype.ofFinite _
      ext x
      obtain ⟨⟨j⟩, a, ha : F.map _ a = x⟩ := Limits.FintypeCat.jointly_surjective
        (Discrete.functor (ff ⟨X, XinI⟩) ⋙ F) _ (Limits.isColimitOfPreserves F (h4 ⟨X, XinI⟩)) x
      rw [FintypeCat.id_apply, ← ha, FunctorToFintypeCat.naturality]
      simp [h ⟨(ff _) j, ⟨Set.range (ff ⟨X, XinI⟩), ⟨⟨_, rfl⟩, ⟨j, rfl⟩⟩⟩⟩]
    /-
      case intro.intro.intro.intro.intro.refine_2
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.GaloisCategory C
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      U : Set ((X : C) → CategoryTheory.Aut (F.obj X))
      hUopen : IsOpen U
      h1 : Membership.mem (Set.preimage (⇑(CategoryTheory.PreGaloisCategory.autEmbed …
      h✝¹ : IsOpen (Set.preimage (⇑(CategoryTheory.PreGaloisCategory.autEmbedding F) …
      I : Finset C
      u : (a : C) → Set (CategoryTheory.Aut (F.obj a))
      ho : ∀ (a : C), Membership.mem I a → And (IsOpen (u a)) (Membership.mem (u a)  …
      ha : HasSubset.Subset ((↑I).pi u) U
      fι : (Subtype fun x => Membership.mem I x) → Type
      ff : (X : Subtype fun x => Membership.mem I x) → fι X → C
      fc : (X : Subtype fun x => Membership.mem I x) → (i : fι X) → Quiver.Hom (ff X …
      h4 : (X : Subtype fun x => Membership.mem I x) → CategoryTheory.Limits.IsColim …
      h5 : ∀ (X : Subtype fun x => Membership.mem I x) (i : fι X), CategoryTheory.Pr …
      h6 : ∀ (X : Subtype fun x => Membership.mem I x), Finite (fι X)
      σ : CategoryTheory.Aut F
      h✝ : ∀ (X : ↑(Set.iUnion fun X => Set.range (ff X))), Eq (σ.hom.app ↑X) (Categ …
      X : C
      XinI : Membership.mem (↑I) X
      h : Eq (σ.hom.app X) (CategoryTheory.CategoryStruct.id (F.obj X))
      ⊢ Eq ((CategoryTheory.PreGaloisCategory.autEmbedding F) σ X) 1
    -/
    exact Iso.ext h
    /-
      🎉 no goals
    -/


/-- The stabilizers of points in the fibers of Galois objects form a neighbourhood basis
of the identity in `Aut F`. -/
lemma nhds_one_has_basis_stabilizers : (nhds (1 : Aut F)).HasBasis (fun _ ↦ True)
    (fun X : PointedGaloisObject F ↦ MulAction.stabilizer (Aut F) X.pt) where
  mem_iff' S := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.GaloisCategory C
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      S : Set (CategoryTheory.Aut F)
      ⊢ Iff (Membership.mem (nhds 1) S) (Exists fun i => And True (HasSubset.Subset  …
    -/
    rw [mem_nhds_iff]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.GaloisCategory C
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      S : Set (CategoryTheory.Aut F)
      ⊢ Iff (Exists fun t => And (HasSubset.Subset t S) (And (IsOpen t) (Membership. …
    -/
    refine ⟨?_, ?_⟩
      /-
        case refine_1
        C : Type u₁
        inst✝² : CategoryTheory.Category.{u₂, u₁} C
        F : CategoryTheory.Functor C FintypeCat
        inst✝¹ : CategoryTheory.GaloisCategory C
        inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
        S : Set (CategoryTheory.Aut F)
        ⊢ (Exists fun t => And (HasSubset.Subset t S) (And (IsOpen t) (Membership.mem  …
      -/
    · intro ⟨U, hU, hUopen, hUone⟩
      /-
        case refine_1
        C : Type u₁
        inst✝² : CategoryTheory.Category.{u₂, u₁} C
        F : CategoryTheory.Functor C FintypeCat
        inst✝¹ : CategoryTheory.GaloisCategory C
        inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
        S U : Set (CategoryTheory.Aut F)
        hU : HasSubset.Subset U S
        hUopen : IsOpen U
        hUone : Membership.mem U 1
        ⊢ Exists fun i => And True (HasSubset.Subset (↑(MulAction.stabilizer (Category …
      -/
      obtain ⟨I, _, hc, hmem⟩ := exists_set_ker_evaluation_subset_of_isOpen F hUone hUopen
      /-
        case refine_1.intro.intro.intro
        C : Type u₁
        inst✝² : CategoryTheory.Category.{u₂, u₁} C
        F : CategoryTheory.Functor C FintypeCat
        inst✝¹ : CategoryTheory.GaloisCategory C
        inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
        S U : Set (CategoryTheory.Aut F)
        hU : HasSubset.Subset U S
        hUopen : IsOpen U
        hUone : Membership.mem U 1
        I : Set C
        w✝ : Fintype ↑I
        hc : ∀ (X : C), Membership.mem I X → CategoryTheory.PreGaloisCategory.IsConnec …
        hmem : ∀ (σ : CategoryTheory.Aut F), (∀ (X : ↑I), Eq (σ.hom.app ↑X) (CategoryT …
        ⊢ Exists fun i => And True (HasSubset.Subset (↑(MulAction.stabilizer (Category …
      -/
      let P : C := ∏ᶜ fun X : I ↦ X.val
      /-
        case refine_1.intro.intro.intro
        C : Type u₁
        inst✝² : CategoryTheory.Category.{u₂, u₁} C
        F : CategoryTheory.Functor C FintypeCat
        inst✝¹ : CategoryTheory.GaloisCategory C
        inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
        S U : Set (CategoryTheory.Aut F)
        hU : HasSubset.Subset U S
        hUopen : IsOpen U
        hUone : Membership.mem U 1
        I : Set C
        w✝ : Fintype ↑I
        hc : ∀ (X : C), Membership.mem I X → CategoryTheory.PreGaloisCategory.IsConnec …
        hmem : ∀ (σ : CategoryTheory.Aut F), (∀ (X : ↑I), Eq (σ.hom.app ↑X) (CategoryT …
        P : C := CategoryTheory.Limits.piObj fun X => ↑X
        ⊢ Exists fun i => And True (HasSubset.Subset (↑(MulAction.stabilizer (Category …
      -/
      obtain ⟨A, a, hgal, hbij⟩ := exists_galois_representative F P
      /-
        case refine_1.intro.intro.intro.intro.intro.intro
        C : Type u₁
        inst✝² : CategoryTheory.Category.{u₂, u₁} C
        F : CategoryTheory.Functor C FintypeCat
        inst✝¹ : CategoryTheory.GaloisCategory C
        inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
        S U : Set (CategoryTheory.Aut F)
        hU : HasSubset.Subset U S
        hUopen : IsOpen U
        hUone : Membership.mem U 1
        I : Set C
        w✝ : Fintype ↑I
        hc : ∀ (X : C), Membership.mem I X → CategoryTheory.PreGaloisCategory.IsConnec …
        hmem : ∀ (σ : CategoryTheory.Aut F), (∀ (X : ↑I), Eq (σ.hom.app ↑X) (CategoryT …
        P : C := CategoryTheory.Limits.piObj fun X => ↑X
        A : C
        a : ↑(F.obj A)
        hgal : CategoryTheory.PreGaloisCategory.IsGalois A
        hbij : Function.Bijective fun f => F.map f a
        ⊢ Exists fun i => And True (HasSubset.Subset (↑(MulAction.stabilizer (Category …
      -/
      refine ⟨⟨A, a, hgal⟩, trivial, ?_⟩
      /-
        case refine_1.intro.intro.intro.intro.intro.intro
        C : Type u₁
        inst✝² : CategoryTheory.Category.{u₂, u₁} C
        F : CategoryTheory.Functor C FintypeCat
        inst✝¹ : CategoryTheory.GaloisCategory C
        inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
        S U : Set (CategoryTheory.Aut F)
        hU : HasSubset.Subset U S
        hUopen : IsOpen U
        hUone : Membership.mem U 1
        I : Set C
        w✝ : Fintype ↑I
        hc : ∀ (X : C), Membership.mem I X → CategoryTheory.PreGaloisCategory.IsConnec …
        hmem : ∀ (σ : CategoryTheory.Aut F), (∀ (X : ↑I), Eq (σ.hom.app ↑X) (CategoryT …
        P : C := CategoryTheory.Limits.piObj fun X => ↑X
        A : C
        a : ↑(F.obj A)
        hgal : CategoryTheory.PreGaloisCategory.IsGalois A
        hbij : Function.Bijective fun f => F.map f a
        ⊢ HasSubset.Subset (↑(MulAction.stabilizer (CategoryTheory.Aut F) { obj := A,  …
      -/
      intro t (ht : t.hom.app A a = a)
      /-
        case refine_1.intro.intro.intro.intro.intro.intro
        C : Type u₁
        inst✝² : CategoryTheory.Category.{u₂, u₁} C
        F : CategoryTheory.Functor C FintypeCat
        inst✝¹ : CategoryTheory.GaloisCategory C
        inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
        S U : Set (CategoryTheory.Aut F)
        hU : HasSubset.Subset U S
        hUopen : IsOpen U
        hUone : Membership.mem U 1
        I : Set C
        w✝ : Fintype ↑I
        hc : ∀ (X : C), Membership.mem I X → CategoryTheory.PreGaloisCategory.IsConnec …
        hmem : ∀ (σ : CategoryTheory.Aut F), (∀ (X : ↑I), Eq (σ.hom.app ↑X) (CategoryT …
        P : C := CategoryTheory.Limits.piObj fun X => ↑X
        A : C
        a : ↑(F.obj A)
        hgal : CategoryTheory.PreGaloisCategory.IsGalois A
        hbij : Function.Bijective fun f => F.map f a
        t : CategoryTheory.Aut F
        ht : Eq (t.hom.app A a) a
        ⊢ Membership.mem S t
      -/
      apply hU
      /-
        case refine_1.intro.intro.intro.intro.intro.intro.a
        C : Type u₁
        inst✝² : CategoryTheory.Category.{u₂, u₁} C
        F : CategoryTheory.Functor C FintypeCat
        inst✝¹ : CategoryTheory.GaloisCategory C
        inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
        S U : Set (CategoryTheory.Aut F)
        hU : HasSubset.Subset U S
        hUopen : IsOpen U
        hUone : Membership.mem U 1
        I : Set C
        w✝ : Fintype ↑I
        hc : ∀ (X : C), Membership.mem I X → CategoryTheory.PreGaloisCategory.IsConnec …
        hmem : ∀ (σ : CategoryTheory.Aut F), (∀ (X : ↑I), Eq (σ.hom.app ↑X) (CategoryT …
        P : C := CategoryTheory.Limits.piObj fun X => ↑X
        A : C
        a : ↑(F.obj A)
        hgal : CategoryTheory.PreGaloisCategory.IsGalois A
        hbij : Function.Bijective fun f => F.map f a
        t : CategoryTheory.Aut F
        ht : Eq (t.hom.app A a) a
        ⊢ Membership.mem U t
      -/
      apply hmem
      /-
        case refine_1.intro.intro.intro.intro.intro.intro.a.a
        C : Type u₁
        inst✝² : CategoryTheory.Category.{u₂, u₁} C
        F : CategoryTheory.Functor C FintypeCat
        inst✝¹ : CategoryTheory.GaloisCategory C
        inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
        S U : Set (CategoryTheory.Aut F)
        hU : HasSubset.Subset U S
        hUopen : IsOpen U
        hUone : Membership.mem U 1
        I : Set C
        w✝ : Fintype ↑I
        hc : ∀ (X : C), Membership.mem I X → CategoryTheory.PreGaloisCategory.IsConnec …
        hmem : ∀ (σ : CategoryTheory.Aut F), (∀ (X : ↑I), Eq (σ.hom.app ↑X) (CategoryT …
        P : C := CategoryTheory.Limits.piObj fun X => ↑X
        A : C
        a : ↑(F.obj A)
        hgal : CategoryTheory.PreGaloisCategory.IsGalois A
        hbij : Function.Bijective fun f => F.map f a
        t : CategoryTheory.Aut F
        ht : Eq (t.hom.app A a) a
        ⊢ ∀ (X : ↑I), Eq (t.hom.app ↑X) (CategoryTheory.CategoryStruct.id (F.obj ↑X))
      -/
      haveI (X : I) : IsConnected X.val := hc X.val X.property
      /-
        case refine_1.intro.intro.intro.intro.intro.intro.a.a
        C : Type u₁
        inst✝² : CategoryTheory.Category.{u₂, u₁} C
        F : CategoryTheory.Functor C FintypeCat
        inst✝¹ : CategoryTheory.GaloisCategory C
        inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
        S U : Set (CategoryTheory.Aut F)
        hU : HasSubset.Subset U S
        hUopen : IsOpen U
        hUone : Membership.mem U 1
        I : Set C
        w✝ : Fintype ↑I
        hc : ∀ (X : C), Membership.mem I X → CategoryTheory.PreGaloisCategory.IsConnec …
        hmem : ∀ (σ : CategoryTheory.Aut F), (∀ (X : ↑I), Eq (σ.hom.app ↑X) (CategoryT …
        P : C := CategoryTheory.Limits.piObj fun X => ↑X
        A : C
        a : ↑(F.obj A)
        hgal : CategoryTheory.PreGaloisCategory.IsGalois A
        hbij : Function.Bijective fun f => F.map f a
        t : CategoryTheory.Aut F
        ht : Eq (t.hom.app A a) a
        this : ∀ (X : ↑I), CategoryTheory.PreGaloisCategory.IsConnected ↑X
        ⊢ ∀ (X : ↑I), Eq (t.hom.app ↑X) (CategoryTheory.CategoryStruct.id (F.obj ↑X))
      -/
      haveI (X : I) : Nonempty (F.obj X.val) := nonempty_fiber_of_isConnected F X
      /-
        case refine_1.intro.intro.intro.intro.intro.intro.a.a
        C : Type u₁
        inst✝² : CategoryTheory.Category.{u₂, u₁} C
        F : CategoryTheory.Functor C FintypeCat
        inst✝¹ : CategoryTheory.GaloisCategory C
        inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
        S U : Set (CategoryTheory.Aut F)
        hU : HasSubset.Subset U S
        hUopen : IsOpen U
        hUone : Membership.mem U 1
        I : Set C
        w✝ : Fintype ↑I
        hc : ∀ (X : C), Membership.mem I X → CategoryTheory.PreGaloisCategory.IsConnec …
        hmem : ∀ (σ : CategoryTheory.Aut F), (∀ (X : ↑I), Eq (σ.hom.app ↑X) (CategoryT …
        P : C := CategoryTheory.Limits.piObj fun X => ↑X
        A : C
        a : ↑(F.obj A)
        hgal : CategoryTheory.PreGaloisCategory.IsGalois A
        hbij : Function.Bijective fun f => F.map f a
        t : CategoryTheory.Aut F
        ht : Eq (t.hom.app A a) a
        this✝ : ∀ (X : ↑I), CategoryTheory.PreGaloisCategory.IsConnected ↑X
        this : ∀ (X : ↑I), Nonempty ↑(F.obj ↑X)
        ⊢ ∀ (X : ↑I), Eq (t.hom.app ↑X) (CategoryTheory.CategoryStruct.id (F.obj ↑X))
      -/
      intro X
      /-
        case refine_1.intro.intro.intro.intro.intro.intro.a.a
        C : Type u₁
        inst✝² : CategoryTheory.Category.{u₂, u₁} C
        F : CategoryTheory.Functor C FintypeCat
        inst✝¹ : CategoryTheory.GaloisCategory C
        inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
        S U : Set (CategoryTheory.Aut F)
        hU : HasSubset.Subset U S
        hUopen : IsOpen U
        hUone : Membership.mem U 1
        I : Set C
        w✝ : Fintype ↑I
        hc : ∀ (X : C), Membership.mem I X → CategoryTheory.PreGaloisCategory.IsConnec …
        hmem : ∀ (σ : CategoryTheory.Aut F), (∀ (X : ↑I), Eq (σ.hom.app ↑X) (CategoryT …
        P : C := CategoryTheory.Limits.piObj fun X => ↑X
        A : C
        a : ↑(F.obj A)
        hgal : CategoryTheory.PreGaloisCategory.IsGalois A
        hbij : Function.Bijective fun f => F.map f a
        t : CategoryTheory.Aut F
        ht : Eq (t.hom.app A a) a
        this✝ : ∀ (X : ↑I), CategoryTheory.PreGaloisCategory.IsConnected ↑X
        this : ∀ (X : ↑I), Nonempty ↑(F.obj ↑X)
        X : ↑I
        ⊢ Eq (t.hom.app ↑X) (CategoryTheory.CategoryStruct.id (F.obj ↑X))
      -/
      ext x
      /-
        case refine_1.intro.intro.intro.intro.intro.intro.a.a.h
        C : Type u₁
        inst✝² : CategoryTheory.Category.{u₂, u₁} C
        F : CategoryTheory.Functor C FintypeCat
        inst✝¹ : CategoryTheory.GaloisCategory C
        inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
        S U : Set (CategoryTheory.Aut F)
        hU : HasSubset.Subset U S
        hUopen : IsOpen U
        hUone : Membership.mem U 1
        I : Set C
        w✝ : Fintype ↑I
        hc : ∀ (X : C), Membership.mem I X → CategoryTheory.PreGaloisCategory.IsConnec …
        hmem : ∀ (σ : CategoryTheory.Aut F), (∀ (X : ↑I), Eq (σ.hom.app ↑X) (CategoryT …
        P : C := CategoryTheory.Limits.piObj fun X => ↑X
        A : C
        a : ↑(F.obj A)
        hgal : CategoryTheory.PreGaloisCategory.IsGalois A
        hbij : Function.Bijective fun f => F.map f a
        t : CategoryTheory.Aut F
        ht : Eq (t.hom.app A a) a
        this✝ : ∀ (X : ↑I), CategoryTheory.PreGaloisCategory.IsConnected ↑X
        this : ∀ (X : ↑I), Nonempty ↑(F.obj ↑X)
        X : ↑I
        x : ↑(F.obj ↑X)
        ⊢ Eq (t.hom.app (↑X) x) (CategoryTheory.CategoryStruct.id (F.obj ↑X) x)
      -/
      simp only [FintypeCat.id_apply]
      obtain ⟨z, rfl⟩ :=
        surjective_of_nonempty_fiber_of_isConnected F (Pi.π (fun X : I ↦ X.val) X) x
      /-
        case refine_1.intro.intro.intro.intro.intro.intro.a.a.h.intro
        C : Type u₁
        inst✝² : CategoryTheory.Category.{u₂, u₁} C
        F : CategoryTheory.Functor C FintypeCat
        inst✝¹ : CategoryTheory.GaloisCategory C
        inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
        S U : Set (CategoryTheory.Aut F)
        hU : HasSubset.Subset U S
        hUopen : IsOpen U
        hUone : Membership.mem U 1
        I : Set C
        w✝ : Fintype ↑I
        hc : ∀ (X : C), Membership.mem I X → CategoryTheory.PreGaloisCategory.IsConnec …
        hmem : ∀ (σ : CategoryTheory.Aut F), (∀ (X : ↑I), Eq (σ.hom.app ↑X) (CategoryT …
        P : C := CategoryTheory.Limits.piObj fun X => ↑X
        A : C
        a : ↑(F.obj A)
        hgal : CategoryTheory.PreGaloisCategory.IsGalois A
        hbij : Function.Bijective fun f => F.map f a
        t : CategoryTheory.Aut F
        ht : Eq (t.hom.app A a) a
        this✝ : ∀ (X : ↑I), CategoryTheory.PreGaloisCategory.IsConnected ↑X
        this : ∀ (X : ↑I), Nonempty ↑(F.obj ↑X)
        X : ↑I
        z : ↑(F.obj (CategoryTheory.Limits.piObj fun X => ↑X))
        ⊢ Eq (t.hom.app (↑X) (F.map (CategoryTheory.Limits.Pi.π (fun X => ↑X) X) z)) ( …
      -/
      obtain ⟨f, rfl⟩ := hbij.surjective z
      /-
        case refine_1.intro.intro.intro.intro.intro.intro.a.a.h.intro.intro
        C : Type u₁
        inst✝² : CategoryTheory.Category.{u₂, u₁} C
        F : CategoryTheory.Functor C FintypeCat
        inst✝¹ : CategoryTheory.GaloisCategory C
        inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
        S U : Set (CategoryTheory.Aut F)
        hU : HasSubset.Subset U S
        hUopen : IsOpen U
        hUone : Membership.mem U 1
        I : Set C
        w✝ : Fintype ↑I
        hc : ∀ (X : C), Membership.mem I X → CategoryTheory.PreGaloisCategory.IsConnec …
        hmem : ∀ (σ : CategoryTheory.Aut F), (∀ (X : ↑I), Eq (σ.hom.app ↑X) (CategoryT …
        P : C := CategoryTheory.Limits.piObj fun X => ↑X
        A : C
        a : ↑(F.obj A)
        hgal : CategoryTheory.PreGaloisCategory.IsGalois A
        hbij : Function.Bijective fun f => F.map f a
        t : CategoryTheory.Aut F
        ht : Eq (t.hom.app A a) a
        this✝ : ∀ (X : ↑I), CategoryTheory.PreGaloisCategory.IsConnected ↑X
        this : ∀ (X : ↑I), Nonempty ↑(F.obj ↑X)
        X : ↑I
        f : Quiver.Hom A P
        ⊢ Eq (t.hom.app (↑X) (F.map (CategoryTheory.Limits.Pi.π (fun X => ↑X) X) ((fun …
      -/
      rw [FunctorToFintypeCat.naturality, FunctorToFintypeCat.naturality, ht]
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        C : Type u₁
        inst✝² : CategoryTheory.Category.{u₂, u₁} C
        F : CategoryTheory.Functor C FintypeCat
        inst✝¹ : CategoryTheory.GaloisCategory C
        inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
        S : Set (CategoryTheory.Aut F)
        ⊢ (Exists fun i => And True (HasSubset.Subset (↑(MulAction.stabilizer (Categor …
      -/
    · intro ⟨X, _, h⟩
      exact ⟨MulAction.stabilizer (Aut F) X.pt, h, stabilizer_isOpen (Aut F) X.pt,
        Subgroup.one_mem _⟩


