/-- We say `G` acts naturally on the fibers of `F` if for every `f : X ⟶ Y`, the `G`-actions
on `F.obj X` and `F.obj Y` are compatible with `F.map f`. -/
class IsNaturalSMul : Prop where
  naturality (g : G) {X Y : C} (f : X ⟶ Y) (x : F.obj X) : F.map f (g • x) = g • F.map f x


variable {G} in
@[simps!]
private def isoOnObj (g : G) (X : C) : F.obj X ≅ F.obj X :=
  FintypeCat.equivEquivIso <| {
    toFun := fun x ↦ g • x
    invFun := fun x ↦ g⁻¹ • x
                           /-
                             C : Type u₁
                             inst✝² : CategoryTheory.Category.{u₂, u₁} C
                             F : CategoryTheory.Functor C FintypeCat
                             G : Type u_1
                             inst✝¹ : Group G
                             inst✝ : (X : C) → MulAction G ↑(F.obj X)
                             g : G
                             X : C
                             x✝ : ↑(F.obj X)
                             ⊢ Eq ((fun x => HSMul.hSMul (Inv.inv g) x) ((fun x => HSMul.hSMul g x) x✝)) x✝
                           -/
    left_inv := fun _ ↦ by simp
                           /-
                             🎉 no goals
                           -/
                            /-
                              C : Type u₁
                              inst✝² : CategoryTheory.Category.{u₂, u₁} C
                              F : CategoryTheory.Functor C FintypeCat
                              G : Type u_1
                              inst✝¹ : Group G
                              inst✝ : (X : C) → MulAction G ↑(F.obj X)
                              g : G
                              X : C
                              x✝ : ↑(F.obj X)
                              ⊢ Eq ((fun x => HSMul.hSMul g x) ((fun x => HSMul.hSMul (Inv.inv g) x) x✝)) x✝
                            -/
    right_inv := fun _ ↦ by simp
                            /-
                              🎉 no goals
                            -/
  }


/-- If `G` acts naturally on `F.obj X` for each `X : C`, this is the canonical
group homomorphism into the automorphism group of `F`. -/
def toAut : G →* Aut F where
  toFun g := NatIso.ofComponents (isoOnObj F g) <| by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : (X : C) → MulAction G ↑(F.obj X)
      inst✝ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
      g : G
      ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp (F. …
    -/
    intro X Y f
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : (X : C) → MulAction G ↑(F.obj X)
      inst✝ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
      g : G
      X Y : C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) (CategoryTheory.PreGaloisCa …
    -/
    ext
    /-
      case h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : (X : C) → MulAction G ↑(F.obj X)
      inst✝ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
      g : G
      X Y : C
      f : Quiver.Hom X Y
      x✝ : ↑(F.obj X)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) (CategoryTheory.PreGaloisCa …
    -/
    simp [IsNaturalSMul.naturality]
    /-
      🎉 no goals
    -/
  map_one' := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : (X : C) → MulAction G ↑(F.obj X)
      inst✝ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
      ⊢ Eq ((fun g => CategoryTheory.NatIso.ofComponents (CategoryTheory.PreGaloisCa …
    -/
    ext
    /-
      case h.w.h.h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : (X : C) → MulAction G ↑(F.obj X)
      inst✝ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
      x✝¹ : C
      x✝ : ↑(F.obj x✝¹)
      ⊢ Eq (((fun g => CategoryTheory.NatIso.ofComponents (CategoryTheory.PreGaloisC …
    -/
    simp only [NatIso.ofComponents_hom_app, isoOnObj_hom, one_smul]
    /-
      case h.w.h.h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : (X : C) → MulAction G ↑(F.obj X)
      inst✝ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
      x✝¹ : C
      x✝ : ↑(F.obj x✝¹)
      ⊢ Eq x✝ ((CategoryTheory.Iso.hom 1).app x✝¹ x✝)
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_mul' := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : (X : C) → MulAction G ↑(F.obj X)
      inst✝ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
      ⊢ ∀ (x y : G), Eq ({ toFun := fun g => CategoryTheory.NatIso.ofComponents (Cat …
    -/
    intro g h
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : (X : C) → MulAction G ↑(F.obj X)
      inst✝ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
      g h : G
      ⊢ Eq ({ toFun := fun g => CategoryTheory.NatIso.ofComponents (CategoryTheory.P …
    -/
    ext X x
    /-
      case h.w.h.h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : (X : C) → MulAction G ↑(F.obj X)
      inst✝ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
      g h : G
      X : C
      x : ↑(F.obj X)
      ⊢ Eq (({ toFun := fun g => CategoryTheory.NatIso.ofComponents (CategoryTheory. …
    -/
    simp only [NatIso.ofComponents_hom_app, isoOnObj_hom, mul_smul]
    /-
      case h.w.h.h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : (X : C) → MulAction G ↑(F.obj X)
      inst✝ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
      g h : G
      X : C
      x : ↑(F.obj X)
      ⊢ Eq (HSMul.hSMul g (HSMul.hSMul h x)) ((HMul.hMul (CategoryTheory.NatIso.ofCo …
    -/
    rfl
    /-
      🎉 no goals
    -/


variable {G} in
@[simp]
lemma toAut_hom_app_apply (g : G) {X : C} (x : F.obj X) : (toAut F G g).hom.app X x = g • x :=
  rfl


/-- `toAut` is injective, if only the identity acts trivially on every fiber. -/
lemma toAut_injective_of_non_trivial (h : ∀ (g : G), (∀ (X : C) (x : F.obj X), g • x = x) → g = 1) :
    Function.Injective (toAut F G) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : (X : C) → MulAction G ↑(F.obj X)
    inst✝ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    h : ∀ (g : G), (∀ (X : C) (x : ↑(F.obj X)), Eq (HSMul.hSMul g x) x) → Eq g 1
    ⊢ Function.Injective ⇑(CategoryTheory.PreGaloisCategory.toAut F G)
  -/
  rw [← MonoidHom.ker_eq_bot_iff, eq_bot_iff]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : (X : C) → MulAction G ↑(F.obj X)
    inst✝ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    h : ∀ (g : G), (∀ (X : C) (x : ↑(F.obj X)), Eq (HSMul.hSMul g x) x) → Eq g 1
    ⊢ LE.le (CategoryTheory.PreGaloisCategory.toAut F G).ker Bot.bot
  -/
  intro g (hg : toAut F G g = 1)
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : (X : C) → MulAction G ↑(F.obj X)
    inst✝ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    h : ∀ (g : G), (∀ (X : C) (x : ↑(F.obj X)), Eq (HSMul.hSMul g x) x) → Eq g 1
    g : G
    hg : Eq ((CategoryTheory.PreGaloisCategory.toAut F G) g) 1
    ⊢ Membership.mem Bot.bot g
  -/
  refine h g (fun X x ↦ ?_)
  have : (toAut F G g).hom.app X = 𝟙 (F.obj X) := by
    rw [hg]
    rfl
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : (X : C) → MulAction G ↑(F.obj X)
    inst✝ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    h : ∀ (g : G), (∀ (X : C) (x : ↑(F.obj X)), Eq (HSMul.hSMul g x) x) → Eq g 1
    g : G
    hg : Eq ((CategoryTheory.PreGaloisCategory.toAut F G) g) 1
    X : C
    x : ↑(F.obj X)
    this : Eq (((CategoryTheory.PreGaloisCategory.toAut F G) g).hom.app X) (Catego …
    ⊢ Eq (HSMul.hSMul g x) x
  -/
  rw [← toAut_hom_app_apply, this, FintypeCat.id_apply]
  /-
    🎉 no goals
  -/


lemma toAut_continuous [TopologicalSpace G] [TopologicalGroup G]
    [∀ (X : C), ContinuousSMul G (F.obj X)] :
    Continuous (toAut F G) := by
  /-
    C : Type u₁
    inst✝⁸ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁷ : Group G
    inst✝⁶ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁵ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝⁴ : CategoryTheory.GaloisCategory C
    inst✝³ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalGroup G
    inst✝ : ∀ (X : C), ContinuousSMul G ↑(F.obj X)
    ⊢ Continuous ⇑(CategoryTheory.PreGaloisCategory.toAut F G)
  -/
  apply continuous_of_continuousAt_one
  /-
    case hf
    C : Type u₁
    inst✝⁸ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁷ : Group G
    inst✝⁶ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁵ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝⁴ : CategoryTheory.GaloisCategory C
    inst✝³ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalGroup G
    inst✝ : ∀ (X : C), ContinuousSMul G ↑(F.obj X)
    ⊢ ContinuousAt (⇑(CategoryTheory.PreGaloisCategory.toAut F G)) 1
  -/
  rw [continuousAt_def, map_one]
  /-
    case hf
    C : Type u₁
    inst✝⁸ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁷ : Group G
    inst✝⁶ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁵ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝⁴ : CategoryTheory.GaloisCategory C
    inst✝³ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalGroup G
    inst✝ : ∀ (X : C), ContinuousSMul G ↑(F.obj X)
    ⊢ ∀ (A : Set (CategoryTheory.Aut F)), Membership.mem (nhds 1) A → Membership.m …
  -/
  intro A hA
  /-
    case hf
    C : Type u₁
    inst✝⁸ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁷ : Group G
    inst✝⁶ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁵ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝⁴ : CategoryTheory.GaloisCategory C
    inst✝³ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalGroup G
    inst✝ : ∀ (X : C), ContinuousSMul G ↑(F.obj X)
    A : Set (CategoryTheory.Aut F)
    hA : Membership.mem (nhds 1) A
    ⊢ Membership.mem (nhds 1) (Set.preimage (⇑(CategoryTheory.PreGaloisCategory.to …
  -/
  obtain ⟨X, _, hX⟩ := ((nhds_one_has_basis_stabilizers F).mem_iff' A).mp hA
  /-
    case hf.intro.intro
    C : Type u₁
    inst✝⁸ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁷ : Group G
    inst✝⁶ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁵ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝⁴ : CategoryTheory.GaloisCategory C
    inst✝³ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalGroup G
    inst✝ : ∀ (X : C), ContinuousSMul G ↑(F.obj X)
    A : Set (CategoryTheory.Aut F)
    hA : Membership.mem (nhds 1) A
    X : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
    left✝ : True
    hX : HasSubset.Subset (↑(MulAction.stabilizer (CategoryTheory.Aut F) X.pt)) A
    ⊢ Membership.mem (nhds 1) (Set.preimage (⇑(CategoryTheory.PreGaloisCategory.to …
  -/
  rw [mem_nhds_iff]
  exact ⟨MulAction.stabilizer G X.pt, Set.preimage_mono (f := toAut F G) hX,
    stabilizer_isOpen G X.pt, one_mem _⟩


lemma action_ext_of_isGalois {t : F ⟶ F} {X : C} [IsGalois X] {g : G} (x : F.obj X)
    (hg : g • x = t.app X x) (y : F.obj X) : g • y = t.app X y := by
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁵ : Group G
    inst✝⁴ : (X : C) → MulAction G ↑(F.obj X)
    inst✝³ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝² : CategoryTheory.GaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    t : Quiver.Hom F F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois X
    g : G
    x : ↑(F.obj X)
    hg : Eq (HSMul.hSMul g x) (t.app X x)
    y : ↑(F.obj X)
    ⊢ Eq (HSMul.hSMul g y) (t.app X y)
  -/
  obtain ⟨φ, (rfl : F.map φ.hom y = x)⟩ := MulAction.exists_smul_eq (Aut X) y x
  have : Function.Injective (F.map φ.hom) :=
    ConcreteCategory.injective_of_mono_of_preservesPullback (F.map φ.hom)
  /-
    case intro
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁵ : Group G
    inst✝⁴ : (X : C) → MulAction G ↑(F.obj X)
    inst✝³ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝² : CategoryTheory.GaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    t : Quiver.Hom F F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois X
    g : G
    y : ↑(F.obj X)
    φ : CategoryTheory.Aut X
    hg : Eq (HSMul.hSMul g (F.map φ.hom y)) (t.app X (F.map φ.hom y))
    this : Function.Injective (F.map φ.hom)
    ⊢ Eq (HSMul.hSMul g y) (t.app X y)
  -/
  apply this
  /-
    case intro.a
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁵ : Group G
    inst✝⁴ : (X : C) → MulAction G ↑(F.obj X)
    inst✝³ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝² : CategoryTheory.GaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    t : Quiver.Hom F F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois X
    g : G
    y : ↑(F.obj X)
    φ : CategoryTheory.Aut X
    hg : Eq (HSMul.hSMul g (F.map φ.hom y)) (t.app X (F.map φ.hom y))
    this : Function.Injective (F.map φ.hom)
    ⊢ Eq (F.map φ.hom (HSMul.hSMul g y)) (F.map φ.hom (t.app X y))
  -/
  rw [IsNaturalSMul.naturality, hg, FunctorToFintypeCat.naturality]
  /-
    🎉 no goals
  -/


lemma toAut_surjective_isGalois (t : Aut F) (X : C) [IsGalois X]
    [MulAction.IsPretransitive G (F.obj X)] :
    ∃ (g : G), ∀ (x : F.obj X), g • x = t.hom.app X x := by
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁶ : Group G
    inst✝⁵ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁴ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝³ : CategoryTheory.GaloisCategory C
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    t : CategoryTheory.Aut F
    X : C
    inst✝¹ : CategoryTheory.PreGaloisCategory.IsGalois X
    inst✝ : MulAction.IsPretransitive G ↑(F.obj X)
    ⊢ Exists fun g => ∀ (x : ↑(F.obj X)), Eq (HSMul.hSMul g x) (t.hom.app X x)
  -/
  obtain ⟨a⟩ := nonempty_fiber_of_isConnected F X
  /-
    case intro
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁶ : Group G
    inst✝⁵ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁴ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝³ : CategoryTheory.GaloisCategory C
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    t : CategoryTheory.Aut F
    X : C
    inst✝¹ : CategoryTheory.PreGaloisCategory.IsGalois X
    inst✝ : MulAction.IsPretransitive G ↑(F.obj X)
    a : ↑(F.obj X)
    ⊢ Exists fun g => ∀ (x : ↑(F.obj X)), Eq (HSMul.hSMul g x) (t.hom.app X x)
  -/
  obtain ⟨g, hg⟩ := MulAction.exists_smul_eq G a (t.hom.app X a)
  /-
    case intro.intro
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁶ : Group G
    inst✝⁵ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁴ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝³ : CategoryTheory.GaloisCategory C
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    t : CategoryTheory.Aut F
    X : C
    inst✝¹ : CategoryTheory.PreGaloisCategory.IsGalois X
    inst✝ : MulAction.IsPretransitive G ↑(F.obj X)
    a : ↑(F.obj X)
    g : G
    hg : Eq (HSMul.hSMul g a) (t.hom.app X a)
    ⊢ Exists fun g => ∀ (x : ↑(F.obj X)), Eq (HSMul.hSMul g x) (t.hom.app X x)
  -/
  exact ⟨g, action_ext_of_isGalois F _ hg⟩
  /-
    🎉 no goals
  -/


lemma toAut_surjective_isGalois_finite_family (t : Aut F) {ι : Type*} [Finite ι] (X : ι → C)
    [∀ i, IsGalois (X i)] (h : ∀ (X : C) [IsGalois X], MulAction.IsPretransitive G (F.obj X)) :
    ∃ (g : G), ∀ (i : ι) (x : F.obj (X i)), g • x = t.hom.app (X i) x := by
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁶ : Group G
    inst✝⁵ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁴ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝³ : CategoryTheory.GaloisCategory C
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    t : CategoryTheory.Aut F
    ι : Type u_2
    inst✝¹ : Finite ι
    X : ι → C
    inst✝ : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsGalois (X i)
    h : ∀ (X : C) [inst : CategoryTheory.PreGaloisCategory.IsGalois X], MulAction. …
    ⊢ Exists fun g => ∀ (i : ι) (x : ↑(F.obj (X i))), Eq (HSMul.hSMul g x) (t.hom. …
  -/
  let x (i : ι) : F.obj (X i) := (nonempty_fiber_of_isConnected F (X i)).some
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁶ : Group G
    inst✝⁵ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁴ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝³ : CategoryTheory.GaloisCategory C
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    t : CategoryTheory.Aut F
    ι : Type u_2
    inst✝¹ : Finite ι
    X : ι → C
    inst✝ : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsGalois (X i)
    h : ∀ (X : C) [inst : CategoryTheory.PreGaloisCategory.IsGalois X], MulAction. …
    x : (i : ι) → ↑(F.obj (X i)) := fun i => ⋯.some
    ⊢ Exists fun g => ∀ (i : ι) (x : ↑(F.obj (X i))), Eq (HSMul.hSMul g x) (t.hom. …
  -/
  let P : C := ∏ᶜ X
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁶ : Group G
    inst✝⁵ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁴ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝³ : CategoryTheory.GaloisCategory C
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    t : CategoryTheory.Aut F
    ι : Type u_2
    inst✝¹ : Finite ι
    X : ι → C
    inst✝ : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsGalois (X i)
    h : ∀ (X : C) [inst : CategoryTheory.PreGaloisCategory.IsGalois X], MulAction. …
    x : (i : ι) → ↑(F.obj (X i)) := fun i => ⋯.some
    P : C := CategoryTheory.Limits.piObj X
    ⊢ Exists fun g => ∀ (i : ι) (x : ↑(F.obj (X i))), Eq (HSMul.hSMul g x) (t.hom. …
  -/
  letI : Fintype ι := Fintype.ofFinite ι
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁶ : Group G
    inst✝⁵ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁴ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝³ : CategoryTheory.GaloisCategory C
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    t : CategoryTheory.Aut F
    ι : Type u_2
    inst✝¹ : Finite ι
    X : ι → C
    inst✝ : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsGalois (X i)
    h : ∀ (X : C) [inst : CategoryTheory.PreGaloisCategory.IsGalois X], MulAction. …
    x : (i : ι) → ↑(F.obj (X i)) := fun i => ⋯.some
    P : C := CategoryTheory.Limits.piObj X
    this : Fintype ι := Fintype.ofFinite ι
    ⊢ Exists fun g => ∀ (i : ι) (x : ↑(F.obj (X i))), Eq (HSMul.hSMul g x) (t.hom. …
  -/
  let is₁ : F.obj P ≅ ∏ᶜ fun i ↦ (F.obj (X i)) := PreservesProduct.iso F X
  let is₂ : (∏ᶜ fun i ↦ F.obj (X i) : FintypeCat) ≃ ∀ i, F.obj (X i) :=
    Limits.FintypeCat.productEquiv (fun i ↦ (F.obj (X i)))
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁶ : Group G
    inst✝⁵ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁴ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝³ : CategoryTheory.GaloisCategory C
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    t : CategoryTheory.Aut F
    ι : Type u_2
    inst✝¹ : Finite ι
    X : ι → C
    inst✝ : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsGalois (X i)
    h : ∀ (X : C) [inst : CategoryTheory.PreGaloisCategory.IsGalois X], MulAction. …
    x : (i : ι) → ↑(F.obj (X i)) := fun i => ⋯.some
    P : C := CategoryTheory.Limits.piObj X
    this : Fintype ι := Fintype.ofFinite ι
    is₁ : CategoryTheory.Iso (F.obj P) (CategoryTheory.Limits.piObj fun i => F.obj …
    is₂ : Equiv (↑(CategoryTheory.Limits.piObj fun i => F.obj (X i))) ((i : ι) → ↑ …
    ⊢ Exists fun g => ∀ (i : ι) (x : ↑(F.obj (X i))), Eq (HSMul.hSMul g x) (t.hom. …
  -/
  let px : F.obj P := is₁.inv (is₂.symm x)
  have hpx (i : ι) : F.map (Pi.π X i) px = x i := by
    simp only [px, is₁, is₂, ← piComparison_comp_π, ← PreservesProduct.iso_hom]
    simp only [FintypeCat.comp_apply, FintypeCat.inv_hom_id_apply,
      FintypeCat.productEquiv_symm_comp_π_apply]
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁶ : Group G
    inst✝⁵ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁴ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝³ : CategoryTheory.GaloisCategory C
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    t : CategoryTheory.Aut F
    ι : Type u_2
    inst✝¹ : Finite ι
    X : ι → C
    inst✝ : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsGalois (X i)
    h : ∀ (X : C) [inst : CategoryTheory.PreGaloisCategory.IsGalois X], MulAction. …
    x : (i : ι) → ↑(F.obj (X i)) := fun i => ⋯.some
    P : C := CategoryTheory.Limits.piObj X
    this : Fintype ι := Fintype.ofFinite ι
    is₁ : CategoryTheory.Iso (F.obj P) (CategoryTheory.Limits.piObj fun i => F.obj …
    is₂ : Equiv (↑(CategoryTheory.Limits.piObj fun i => F.obj (X i))) ((i : ι) → ↑ …
    px : ↑(F.obj P) := is₁.inv (is₂.symm x)
    hpx : ∀ (i : ι), Eq (F.map (CategoryTheory.Limits.Pi.π X i) px) (x i)
    ⊢ Exists fun g => ∀ (i : ι) (x : ↑(F.obj (X i))), Eq (HSMul.hSMul g x) (t.hom. …
  -/
  obtain ⟨A, f, a, _, hfa⟩ := exists_hom_from_galois_of_fiber F P px
  /-
    case intro.intro.intro.intro
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁶ : Group G
    inst✝⁵ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁴ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝³ : CategoryTheory.GaloisCategory C
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    t : CategoryTheory.Aut F
    ι : Type u_2
    inst✝¹ : Finite ι
    X : ι → C
    inst✝ : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsGalois (X i)
    h : ∀ (X : C) [inst : CategoryTheory.PreGaloisCategory.IsGalois X], MulAction. …
    x : (i : ι) → ↑(F.obj (X i)) := fun i => ⋯.some
    P : C := CategoryTheory.Limits.piObj X
    this : Fintype ι := Fintype.ofFinite ι
    is₁ : CategoryTheory.Iso (F.obj P) (CategoryTheory.Limits.piObj fun i => F.obj …
    is₂ : Equiv (↑(CategoryTheory.Limits.piObj fun i => F.obj (X i))) ((i : ι) → ↑ …
    px : ↑(F.obj P) := is₁.inv (is₂.symm x)
    hpx : ∀ (i : ι), Eq (F.map (CategoryTheory.Limits.Pi.π X i) px) (x i)
    A : C
    f : Quiver.Hom A P
    a : ↑(F.obj A)
    left✝ : CategoryTheory.PreGaloisCategory.IsGalois A
    hfa : Eq (F.map f a) px
    ⊢ Exists fun g => ∀ (i : ι) (x : ↑(F.obj (X i))), Eq (HSMul.hSMul g x) (t.hom. …
  -/
  obtain ⟨g, hg⟩ := toAut_surjective_isGalois F G t A
  /-
    case intro.intro.intro.intro.intro
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁶ : Group G
    inst✝⁵ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁴ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝³ : CategoryTheory.GaloisCategory C
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    t : CategoryTheory.Aut F
    ι : Type u_2
    inst✝¹ : Finite ι
    X : ι → C
    inst✝ : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsGalois (X i)
    h : ∀ (X : C) [inst : CategoryTheory.PreGaloisCategory.IsGalois X], MulAction. …
    x : (i : ι) → ↑(F.obj (X i)) := fun i => ⋯.some
    P : C := CategoryTheory.Limits.piObj X
    this : Fintype ι := Fintype.ofFinite ι
    is₁ : CategoryTheory.Iso (F.obj P) (CategoryTheory.Limits.piObj fun i => F.obj …
    is₂ : Equiv (↑(CategoryTheory.Limits.piObj fun i => F.obj (X i))) ((i : ι) → ↑ …
    px : ↑(F.obj P) := is₁.inv (is₂.symm x)
    hpx : ∀ (i : ι), Eq (F.map (CategoryTheory.Limits.Pi.π X i) px) (x i)
    A : C
    f : Quiver.Hom A P
    a : ↑(F.obj A)
    left✝ : CategoryTheory.PreGaloisCategory.IsGalois A
    hfa : Eq (F.map f a) px
    g : G
    hg : ∀ (x : ↑(F.obj A)), Eq (HSMul.hSMul g x) (t.hom.app A x)
    ⊢ Exists fun g => ∀ (i : ι) (x : ↑(F.obj (X i))), Eq (HSMul.hSMul g x) (t.hom. …
  -/
  refine ⟨g, fun i y ↦ action_ext_of_isGalois F (x i) ?_ _⟩
  rw [← hpx i, ← IsNaturalSMul.naturality, FunctorToFintypeCat.naturality,
    ← hfa, FunctorToFintypeCat.naturality, ← IsNaturalSMul.naturality, hg]


/-- If `G` is a compact, topological group that acts continuously and naturally on the
fibers of `F`, `toAut F G` is surjective if and only if it acts transitively on the fibers
of all Galois objects. This is the `if` direction. For the `only if` see
`isPretransitive_of_surjective`. -/
lemma toAut_surjective_of_isPretransitive [TopologicalSpace G] [TopologicalGroup G] [CompactSpace G]
    [∀ (X : C), ContinuousSMul G (F.obj X)]
    (h : ∀ (X : C) [IsGalois X], MulAction.IsPretransitive G (F.obj X)) :
    Function.Surjective (toAut F G) := by
  /-
    C : Type u₁
    inst✝⁹ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁶ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝⁵ : CategoryTheory.GaloisCategory C
    inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : CompactSpace G
    inst✝ : ∀ (X : C), ContinuousSMul G ↑(F.obj X)
    h : ∀ (X : C) [inst : CategoryTheory.PreGaloisCategory.IsGalois X], MulAction. …
    ⊢ Function.Surjective ⇑(CategoryTheory.PreGaloisCategory.toAut F G)
  -/
  intro t
  /-
    C : Type u₁
    inst✝⁹ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁶ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝⁵ : CategoryTheory.GaloisCategory C
    inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : CompactSpace G
    inst✝ : ∀ (X : C), ContinuousSMul G ↑(F.obj X)
    h : ∀ (X : C) [inst : CategoryTheory.PreGaloisCategory.IsGalois X], MulAction. …
    t : CategoryTheory.Aut F
    ⊢ Exists fun a => Eq ((CategoryTheory.PreGaloisCategory.toAut F G) a) t
  -/
  choose gi hgi using (fun X : PointedGaloisObject F ↦ toAut_surjective_isGalois F G t X)
  /-
    C : Type u₁
    inst✝⁹ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁶ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝⁵ : CategoryTheory.GaloisCategory C
    inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : CompactSpace G
    inst✝ : ∀ (X : C), ContinuousSMul G ↑(F.obj X)
    h : ∀ (X : C) [inst : CategoryTheory.PreGaloisCategory.IsGalois X], MulAction. …
    t : CategoryTheory.Aut F
    gi : CategoryTheory.PreGaloisCategory.PointedGaloisObject F → G
    hgi : ∀ (X : CategoryTheory.PreGaloisCategory.PointedGaloisObject F) (x : ↑(F. …
    ⊢ Exists fun a => Eq ((CategoryTheory.PreGaloisCategory.toAut F G) a) t
  -/
  let cl (X : PointedGaloisObject F) : Set G := gi X • MulAction.stabilizer G X.pt
  /-
    C : Type u₁
    inst✝⁹ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁶ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝⁵ : CategoryTheory.GaloisCategory C
    inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : CompactSpace G
    inst✝ : ∀ (X : C), ContinuousSMul G ↑(F.obj X)
    h : ∀ (X : C) [inst : CategoryTheory.PreGaloisCategory.IsGalois X], MulAction. …
    t : CategoryTheory.Aut F
    gi : CategoryTheory.PreGaloisCategory.PointedGaloisObject F → G
    hgi : ∀ (X : CategoryTheory.PreGaloisCategory.PointedGaloisObject F) (x : ↑(F. …
    cl : CategoryTheory.PreGaloisCategory.PointedGaloisObject F → Set G := fun X = …
    ⊢ Exists fun a => Eq ((CategoryTheory.PreGaloisCategory.toAut F G) a) t
  -/
  let c : Set G := ⋂ i, cl i
  have hne : c.Nonempty := by
    rw [← Set.univ_inter c]
    apply CompactSpace.isCompact_univ.inter_iInter_nonempty
    · intro X
      apply IsClosed.leftCoset
      exact Subgroup.isClosed_of_isOpen _ (stabilizer_isOpen G X.pt)
    · intro s
      rw [Set.univ_inter]
      obtain ⟨gs, hgs⟩ :=
        toAut_surjective_isGalois_finite_family F G t (fun X : s ↦ X.val.obj) h
      use gs
      simp only [Set.mem_iInter]
      intro X hXmem
      rw [mem_leftCoset_iff, SetLike.mem_coe, MulAction.mem_stabilizer_iff, mul_smul,
        hgs ⟨X, hXmem⟩, ← hgi X, inv_smul_smul]
  /-
    C : Type u₁
    inst✝⁹ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁶ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝⁵ : CategoryTheory.GaloisCategory C
    inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : CompactSpace G
    inst✝ : ∀ (X : C), ContinuousSMul G ↑(F.obj X)
    h : ∀ (X : C) [inst : CategoryTheory.PreGaloisCategory.IsGalois X], MulAction. …
    t : CategoryTheory.Aut F
    gi : CategoryTheory.PreGaloisCategory.PointedGaloisObject F → G
    hgi : ∀ (X : CategoryTheory.PreGaloisCategory.PointedGaloisObject F) (x : ↑(F. …
    cl : CategoryTheory.PreGaloisCategory.PointedGaloisObject F → Set G := fun X = …
    c : Set G := Set.iInter fun i => cl i
    hne : c.Nonempty
    ⊢ Exists fun a => Eq ((CategoryTheory.PreGaloisCategory.toAut F G) a) t
  -/
  obtain ⟨g, hg⟩ := hne
  /-
    case intro
    C : Type u₁
    inst✝⁹ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁶ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝⁵ : CategoryTheory.GaloisCategory C
    inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : CompactSpace G
    inst✝ : ∀ (X : C), ContinuousSMul G ↑(F.obj X)
    h : ∀ (X : C) [inst : CategoryTheory.PreGaloisCategory.IsGalois X], MulAction. …
    t : CategoryTheory.Aut F
    gi : CategoryTheory.PreGaloisCategory.PointedGaloisObject F → G
    hgi : ∀ (X : CategoryTheory.PreGaloisCategory.PointedGaloisObject F) (x : ↑(F. …
    cl : CategoryTheory.PreGaloisCategory.PointedGaloisObject F → Set G := fun X = …
    c : Set G := Set.iInter fun i => cl i
    g : G
    hg : Membership.mem c g
    ⊢ Exists fun a => Eq ((CategoryTheory.PreGaloisCategory.toAut F G) a) t
  -/
  refine ⟨g, Iso.ext <| natTrans_ext_of_isGalois _ <| fun X _ ↦ ?_⟩
  /-
    case intro
    C : Type u₁
    inst✝⁹ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁶ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝⁵ : CategoryTheory.GaloisCategory C
    inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : CompactSpace G
    inst✝ : ∀ (X : C), ContinuousSMul G ↑(F.obj X)
    h : ∀ (X : C) [inst : CategoryTheory.PreGaloisCategory.IsGalois X], MulAction. …
    t : CategoryTheory.Aut F
    gi : CategoryTheory.PreGaloisCategory.PointedGaloisObject F → G
    hgi : ∀ (X : CategoryTheory.PreGaloisCategory.PointedGaloisObject F) (x : ↑(F. …
    cl : CategoryTheory.PreGaloisCategory.PointedGaloisObject F → Set G := fun X = …
    c : Set G := Set.iInter fun i => cl i
    g : G
    hg : Membership.mem c g
    X : C
    x✝ : CategoryTheory.PreGaloisCategory.IsGalois X
    ⊢ Eq (((CategoryTheory.PreGaloisCategory.toAut F G) g).hom.app X) (t.hom.app X)
  -/
  ext x
  /-
    case intro.h
    C : Type u₁
    inst✝⁹ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁶ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝⁵ : CategoryTheory.GaloisCategory C
    inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : CompactSpace G
    inst✝ : ∀ (X : C), ContinuousSMul G ↑(F.obj X)
    h : ∀ (X : C) [inst : CategoryTheory.PreGaloisCategory.IsGalois X], MulAction. …
    t : CategoryTheory.Aut F
    gi : CategoryTheory.PreGaloisCategory.PointedGaloisObject F → G
    hgi : ∀ (X : CategoryTheory.PreGaloisCategory.PointedGaloisObject F) (x : ↑(F. …
    cl : CategoryTheory.PreGaloisCategory.PointedGaloisObject F → Set G := fun X = …
    c : Set G := Set.iInter fun i => cl i
    g : G
    hg : Membership.mem c g
    X : C
    x✝ : CategoryTheory.PreGaloisCategory.IsGalois X
    x : ↑(F.obj X)
    ⊢ Eq (((CategoryTheory.PreGaloisCategory.toAut F G) g).hom.app X x) (t.hom.app …
  -/
  simp only [toAut_hom_app_apply]
  have : g ∈ (gi ⟨X, x, inferInstance⟩ • MulAction.stabilizer G x : Set G) := by
    simp only [Set.mem_iInter, c] at hg
    exact hg _
  /-
    case intro.h
    C : Type u₁
    inst✝⁹ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁶ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝⁵ : CategoryTheory.GaloisCategory C
    inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : CompactSpace G
    inst✝ : ∀ (X : C), ContinuousSMul G ↑(F.obj X)
    h : ∀ (X : C) [inst : CategoryTheory.PreGaloisCategory.IsGalois X], MulAction. …
    t : CategoryTheory.Aut F
    gi : CategoryTheory.PreGaloisCategory.PointedGaloisObject F → G
    hgi : ∀ (X : CategoryTheory.PreGaloisCategory.PointedGaloisObject F) (x : ↑(F. …
    cl : CategoryTheory.PreGaloisCategory.PointedGaloisObject F → Set G := fun X = …
    c : Set G := Set.iInter fun i => cl i
    g : G
    hg : Membership.mem c g
    X : C
    x✝ : CategoryTheory.PreGaloisCategory.IsGalois X
    x : ↑(F.obj X)
    this : Membership.mem (HSMul.hSMul (gi { obj := X, pt := x, isGalois := ⋯ }) ↑ …
    ⊢ Eq (HSMul.hSMul g x) (t.hom.app X x)
  -/
  obtain ⟨s, (hsmem : s • x = x), (rfl : gi ⟨X, x, inferInstance⟩ • s = _)⟩ := this
  /-
    case intro.h.intro.intro
    C : Type u₁
    inst✝⁹ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁶ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝⁵ : CategoryTheory.GaloisCategory C
    inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : CompactSpace G
    inst✝ : ∀ (X : C), ContinuousSMul G ↑(F.obj X)
    h : ∀ (X : C) [inst : CategoryTheory.PreGaloisCategory.IsGalois X], MulAction. …
    t : CategoryTheory.Aut F
    gi : CategoryTheory.PreGaloisCategory.PointedGaloisObject F → G
    hgi : ∀ (X : CategoryTheory.PreGaloisCategory.PointedGaloisObject F) (x : ↑(F. …
    cl : CategoryTheory.PreGaloisCategory.PointedGaloisObject F → Set G := fun X = …
    c : Set G := Set.iInter fun i => cl i
    X : C
    x✝ : CategoryTheory.PreGaloisCategory.IsGalois X
    x : ↑(F.obj X)
    s : G
    hsmem : Eq (HSMul.hSMul s x) x
    hg : Membership.mem c (HSMul.hSMul (gi { obj := X, pt := x, isGalois := ⋯ }) s)
    ⊢ Eq (HSMul.hSMul (HSMul.hSMul (gi { obj := X, pt := x, isGalois := ⋯ }) s) x) …
  -/
  rw [smul_eq_mul, mul_smul, hsmem]
  /-
    case intro.h.intro.intro
    C : Type u₁
    inst✝⁹ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁶ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
    inst✝⁵ : CategoryTheory.GaloisCategory C
    inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : CompactSpace G
    inst✝ : ∀ (X : C), ContinuousSMul G ↑(F.obj X)
    h : ∀ (X : C) [inst : CategoryTheory.PreGaloisCategory.IsGalois X], MulAction. …
    t : CategoryTheory.Aut F
    gi : CategoryTheory.PreGaloisCategory.PointedGaloisObject F → G
    hgi : ∀ (X : CategoryTheory.PreGaloisCategory.PointedGaloisObject F) (x : ↑(F. …
    cl : CategoryTheory.PreGaloisCategory.PointedGaloisObject F → Set G := fun X = …
    c : Set G := Set.iInter fun i => cl i
    X : C
    x✝ : CategoryTheory.PreGaloisCategory.IsGalois X
    x : ↑(F.obj X)
    s : G
    hsmem : Eq (HSMul.hSMul s x) x
    hg : Membership.mem c (HSMul.hSMul (gi { obj := X, pt := x, isGalois := ⋯ }) s)
    ⊢ Eq (HSMul.hSMul (gi { obj := X, pt := x, isGalois := ⋯ }) x) (t.hom.app X x)
  -/
  exact hgi ⟨X, x, inferInstance⟩ x
  /-
    🎉 no goals
  -/


/-- If `toAut F G` is surjective, then `G` acts transitively on the fibers of connected objects.
For a converse see `toAut_surjective`. -/
lemma isPretransitive_of_surjective (h : Function.Surjective (toAut F G)) (X : C)
    [IsConnected X] : MulAction.IsPretransitive G (F.obj X) where
  exists_smul_eq x y := by
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      G : Type u_1
      inst✝⁵ : Group G
      inst✝⁴ : (X : C) → MulAction G ↑(F.obj X)
      inst✝³ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
      inst✝² : CategoryTheory.GaloisCategory C
      inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      h : Function.Surjective ⇑(CategoryTheory.PreGaloisCategory.toAut F G)
      X : C
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
      x y : ↑(F.obj X)
      ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
    -/
    obtain ⟨t, ht⟩ := MulAction.exists_smul_eq (Aut F) x y
    /-
      case intro
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      G : Type u_1
      inst✝⁵ : Group G
      inst✝⁴ : (X : C) → MulAction G ↑(F.obj X)
      inst✝³ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
      inst✝² : CategoryTheory.GaloisCategory C
      inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      h : Function.Surjective ⇑(CategoryTheory.PreGaloisCategory.toAut F G)
      X : C
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
      x y : ↑(F.obj X)
      t : CategoryTheory.Aut F
      ht : Eq (HSMul.hSMul t x) y
      ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
    -/
    obtain ⟨g, rfl⟩ := h t
    /-
      case intro.intro
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      G : Type u_1
      inst✝⁵ : Group G
      inst✝⁴ : (X : C) → MulAction G ↑(F.obj X)
      inst✝³ : CategoryTheory.PreGaloisCategory.IsNaturalSMul F G
      inst✝² : CategoryTheory.GaloisCategory C
      inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      h : Function.Surjective ⇑(CategoryTheory.PreGaloisCategory.toAut F G)
      X : C
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
      x y : ↑(F.obj X)
      g : G
      ht : Eq (HSMul.hSMul ((CategoryTheory.PreGaloisCategory.toAut F G) g) x) y
      ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
    -/
    exact ⟨g, ht⟩
    /-
      🎉 no goals
    -/


/-- A compact, topological group `G` with a natural action on `F.obj X` for each `X : C`
is a fundamental group of `F`, if `G` acts transitively on the fibers of Galois objects,
the action on `F.obj X` is continuous for all `X : C` and the only trivially acting element of `G`
is the identity. -/
class IsFundamentalGroup [TopologicalSpace G] [TopologicalGroup G] [CompactSpace G]
    extends IsNaturalSMul F G : Prop where
  transitive_of_isGalois (X : C) [IsGalois X] : MulAction.IsPretransitive G (F.obj X)
  continuous_smul (X : C) : ContinuousSMul G (F.obj X)
  non_trivial' (g : G) : (∀ (X : C) (x : F.obj X), g • x = x) → g = 1


lemma non_trivial (g : G) (h : ∀ (X : C) (x : F.obj X), g • x = x) : g = 1 :=
  IsFundamentalGroup.non_trivial' g h


/-- `Aut F` is a fundamental group for `F`. -/
instance : IsFundamentalGroup F (Aut F) where
  naturality g _ _ f x := (FunctorToFintypeCat.naturality F F g.hom f x).symm
  transitive_of_isGalois X := FiberFunctor.isPretransitive_of_isConnected F X
  continuous_smul X := continuousSMul_aut_fiber F X
  non_trivial' g h := by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝³ : CategoryTheory.GaloisCategory C
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : (X : C) → MulAction G ↑(F.obj X)
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      g : CategoryTheory.Aut F
      h : ∀ (X : C) (x : ↑(F.obj X)), Eq (HSMul.hSMul g x) x
      ⊢ Eq g 1
    -/
    ext X x
    /-
      case h.w.h.h
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝³ : CategoryTheory.GaloisCategory C
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : (X : C) → MulAction G ↑(F.obj X)
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      g : CategoryTheory.Aut F
      h : ∀ (X : C) (x : ↑(F.obj X)), Eq (HSMul.hSMul g x) x
      X : C
      x : ↑(F.obj X)
      ⊢ Eq (g.hom.app X x) ((CategoryTheory.Iso.hom 1).app X x)
    -/
    exact h X x
    /-
      🎉 no goals
    -/


lemma toAut_bijective : Function.Bijective (toAut F G) where
  left := toAut_injective_of_non_trivial F G IsFundamentalGroup.non_trivial'
  right := toAut_surjective_of_isPretransitive F G IsFundamentalGroup.transitive_of_isGalois


instance (X : C) [IsConnected X] : MulAction.IsPretransitive G (F.obj X) :=
  isPretransitive_of_surjective F G (toAut_bijective F G).surjective X


/-- If `G` is the fundamental group for `F`, it is isomorphic to `Aut F` as groups and
this isomorphism is also a homeomorphism (see `toAutMulEquiv_isHomeomorph`). -/
noncomputable def toAutMulEquiv : G ≃* Aut F :=
  MulEquiv.ofBijective (toAut F G) (toAut_bijective F G)


lemma toAut_isHomeomorph : IsHomeomorph (toAut F G) := by
  /-
    C : Type u₁
    inst✝⁸ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝⁷ : CategoryTheory.GaloisCategory C
    G : Type u_1
    inst✝⁶ : Group G
    inst✝⁵ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : CompactSpace G
    inst✝ : CategoryTheory.PreGaloisCategory.IsFundamentalGroup F G
    ⊢ IsHomeomorph ⇑(CategoryTheory.PreGaloisCategory.toAut F G)
  -/
  rw [isHomeomorph_iff_continuous_bijective]
  /-
    C : Type u₁
    inst✝⁸ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝⁷ : CategoryTheory.GaloisCategory C
    G : Type u_1
    inst✝⁶ : Group G
    inst✝⁵ : (X : C) → MulAction G ↑(F.obj X)
    inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    inst✝³ : TopologicalSpace G
    inst✝² : TopologicalGroup G
    inst✝¹ : CompactSpace G
    inst✝ : CategoryTheory.PreGaloisCategory.IsFundamentalGroup F G
    ⊢ And (Continuous ⇑(CategoryTheory.PreGaloisCategory.toAut F G)) (Function.Bij …
  -/
  exact ⟨toAut_continuous F G, toAut_bijective F G⟩
  /-
    🎉 no goals
  -/


lemma toAutMulEquiv_isHomeomorph : IsHomeomorph (toAutMulEquiv F G) :=
  toAut_isHomeomorph F G


/-- If `G` is a fundamental group for `F`, it is canonically homeomorphic to `Aut F`. -/
noncomputable def toAutHomeo : G ≃ₜ Aut F := (toAut_isHomeomorph F G).homeomorph


@[simp]
lemma toAutMulEquiv_apply (g : G) : toAutMulEquiv F G g = toAut F G g := rfl


@[simp]
lemma toAutHomeo_apply (g : G) : toAutHomeo F G g = toAut F G g := rfl


