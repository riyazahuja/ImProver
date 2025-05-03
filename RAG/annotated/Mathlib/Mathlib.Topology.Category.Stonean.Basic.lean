/-- `Stonean` is the category of extremally disconnected compact Hausdorff spaces. -/
abbrev Stonean := CompHausLike (fun X ↦ ExtremallyDisconnected X)


/-- `Projective` implies `ExtremallyDisconnected`. -/
instance (X : CompHaus.{u}) [Projective X] : ExtremallyDisconnected X := by
  /-
    X : CompHaus
    inst✝ : CategoryTheory.Projective X
    ⊢ ExtremallyDisconnected ↑X.toTop
  -/
  apply CompactT2.Projective.extremallyDisconnected
  /-
    case h
    X : CompHaus
    inst✝ : CategoryTheory.Projective X
    ⊢ CompactT2.Projective ↑X.toTop
  -/
  intro A B _ _ _ _ _ _ f g hf hg hsurj
  /-
    case h
    X : CompHaus
    inst✝⁶ : CategoryTheory.Projective X
    A B : Type u
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : TopologicalSpace B
    inst✝³ : CompactSpace A
    inst✝² : T2Space A
    inst✝¹ : CompactSpace B
    inst✝ : T2Space B
    f : ↑X.toTop → B
    g : A → B
    hf : Continuous f
    hg : Continuous g
    hsurj : Function.Surjective g
    ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp g h) f)
  -/
  let A' : CompHaus := CompHaus.of A
  /-
    case h
    X : CompHaus
    inst✝⁶ : CategoryTheory.Projective X
    A B : Type u
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : TopologicalSpace B
    inst✝³ : CompactSpace A
    inst✝² : T2Space A
    inst✝¹ : CompactSpace B
    inst✝ : T2Space B
    f : ↑X.toTop → B
    g : A → B
    hf : Continuous f
    hg : Continuous g
    hsurj : Function.Surjective g
    A' : CompHaus := CompHaus.of A
    ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp g h) f)
  -/
  let B' : CompHaus := CompHaus.of B
  /-
    case h
    X : CompHaus
    inst✝⁶ : CategoryTheory.Projective X
    A B : Type u
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : TopologicalSpace B
    inst✝³ : CompactSpace A
    inst✝² : T2Space A
    inst✝¹ : CompactSpace B
    inst✝ : T2Space B
    f : ↑X.toTop → B
    g : A → B
    hf : Continuous f
    hg : Continuous g
    hsurj : Function.Surjective g
    A' : CompHaus := CompHaus.of A
    B' : CompHaus := CompHaus.of B
    ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp g h) f)
  -/
  let f' : X ⟶ B' := ⟨f, hf⟩
  /-
    case h
    X : CompHaus
    inst✝⁶ : CategoryTheory.Projective X
    A B : Type u
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : TopologicalSpace B
    inst✝³ : CompactSpace A
    inst✝² : T2Space A
    inst✝¹ : CompactSpace B
    inst✝ : T2Space B
    f : ↑X.toTop → B
    g : A → B
    hf : Continuous f
    hg : Continuous g
    hsurj : Function.Surjective g
    A' : CompHaus := CompHaus.of A
    B' : CompHaus := CompHaus.of B
    f' : Quiver.Hom X B' := { toFun := f, continuous_toFun := hf }
    ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp g h) f)
  -/
  let g' : A' ⟶ B' := ⟨g,hg⟩
  have : Epi g' := by
    rw [CompHaus.epi_iff_surjective]
    assumption
  /-
    case h
    X : CompHaus
    inst✝⁶ : CategoryTheory.Projective X
    A B : Type u
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : TopologicalSpace B
    inst✝³ : CompactSpace A
    inst✝² : T2Space A
    inst✝¹ : CompactSpace B
    inst✝ : T2Space B
    f : ↑X.toTop → B
    g : A → B
    hf : Continuous f
    hg : Continuous g
    hsurj : Function.Surjective g
    A' : CompHaus := CompHaus.of A
    B' : CompHaus := CompHaus.of B
    f' : Quiver.Hom X B' := { toFun := f, continuous_toFun := hf }
    g' : Quiver.Hom A' B' := { toFun := g, continuous_toFun := hg }
    this : CategoryTheory.Epi g'
    ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp g h) f)
  -/
  obtain ⟨h, hh⟩ := Projective.factors f' g'
  /-
    case h.intro
    X : CompHaus
    inst✝⁶ : CategoryTheory.Projective X
    A B : Type u
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : TopologicalSpace B
    inst✝³ : CompactSpace A
    inst✝² : T2Space A
    inst✝¹ : CompactSpace B
    inst✝ : T2Space B
    f : ↑X.toTop → B
    g : A → B
    hf : Continuous f
    hg : Continuous g
    hsurj : Function.Surjective g
    A' : CompHaus := CompHaus.of A
    B' : CompHaus := CompHaus.of B
    f' : Quiver.Hom X B' := { toFun := f, continuous_toFun := hf }
    g' : Quiver.Hom A' B' := { toFun := g, continuous_toFun := hg }
    this : CategoryTheory.Epi g'
    h : Quiver.Hom X A'
    hh : Eq (CategoryTheory.CategoryStruct.comp h g') f'
    ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp g h) f)
  -/
  refine ⟨h, h.2, ?_⟩
  /-
    case h.intro
    X : CompHaus
    inst✝⁶ : CategoryTheory.Projective X
    A B : Type u
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : TopologicalSpace B
    inst✝³ : CompactSpace A
    inst✝² : T2Space A
    inst✝¹ : CompactSpace B
    inst✝ : T2Space B
    f : ↑X.toTop → B
    g : A → B
    hf : Continuous f
    hg : Continuous g
    hsurj : Function.Surjective g
    A' : CompHaus := CompHaus.of A
    B' : CompHaus := CompHaus.of B
    f' : Quiver.Hom X B' := { toFun := f, continuous_toFun := hf }
    g' : Quiver.Hom A' B' := { toFun := g, continuous_toFun := hg }
    this : CategoryTheory.Epi g'
    h : Quiver.Hom X A'
    hh : Eq (CategoryTheory.CategoryStruct.comp h g') f'
    ⊢ Eq (Function.comp g ⇑h) f
  -/
  ext t
  /-
    case h.intro.h
    X : CompHaus
    inst✝⁶ : CategoryTheory.Projective X
    A B : Type u
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : TopologicalSpace B
    inst✝³ : CompactSpace A
    inst✝² : T2Space A
    inst✝¹ : CompactSpace B
    inst✝ : T2Space B
    f : ↑X.toTop → B
    g : A → B
    hf : Continuous f
    hg : Continuous g
    hsurj : Function.Surjective g
    A' : CompHaus := CompHaus.of A
    B' : CompHaus := CompHaus.of B
    f' : Quiver.Hom X B' := { toFun := f, continuous_toFun := hf }
    g' : Quiver.Hom A' B' := { toFun := g, continuous_toFun := hg }
    this : CategoryTheory.Epi g'
    h : Quiver.Hom X A'
    hh : Eq (CategoryTheory.CategoryStruct.comp h g') f'
    t : ↑X.toTop
    ⊢ Eq (Function.comp g (⇑h) t) (f t)
  -/
  apply_fun (fun e => e t) at hh
  /-
    case h.intro.h
    X : CompHaus
    inst✝⁶ : CategoryTheory.Projective X
    A B : Type u
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : TopologicalSpace B
    inst✝³ : CompactSpace A
    inst✝² : T2Space A
    inst✝¹ : CompactSpace B
    inst✝ : T2Space B
    f : ↑X.toTop → B
    g : A → B
    hf : Continuous f
    hg : Continuous g
    hsurj : Function.Surjective g
    A' : CompHaus := CompHaus.of A
    B' : CompHaus := CompHaus.of B
    f' : Quiver.Hom X B' := { toFun := f, continuous_toFun := hf }
    g' : Quiver.Hom A' B' := { toFun := g, continuous_toFun := hg }
    this : CategoryTheory.Epi g'
    h : Quiver.Hom X A'
    t : ↑X.toTop
    hh : Eq ((CategoryTheory.CategoryStruct.comp h g') t) (f' t)
    ⊢ Eq (Function.comp g (⇑h) t) (f t)
  -/
  exact hh
  /-
    🎉 no goals
  -/


/-- `Projective` implies `Stonean`. -/
@[simps!]
def toStonean (X : CompHaus.{u}) [Projective X] :
    Stonean where
  toTop := X.toTop
  prop := inferInstance


/-- The (forgetful) functor from Stonean spaces to compact Hausdorff spaces. -/
abbrev toCompHaus : Stonean.{u} ⥤ CompHaus.{u} :=
  compHausLikeToCompHaus _


/-- The forgetful functor `Stonean ⥤ CompHaus` is fully faithful. -/
abbrev fullyFaithfulToCompHaus : toCompHaus.FullyFaithful  :=
  CompHausLike.fullyFaithfulToCompHausLike _


instance (X : Type*) [TopologicalSpace X]
    [ExtremallyDisconnected X] : HasProp (fun Y ↦ ExtremallyDisconnected Y) X :=
  ⟨(inferInstance : ExtremallyDisconnected X)⟩


/-- Construct a term of `Stonean` from a type endowed with the structure of a
compact, Hausdorff and extremally disconnected topological space.
-/
abbrev of (X : Type*) [TopologicalSpace X] [CompactSpace X] [T2Space X]
    [ExtremallyDisconnected X] : Stonean := CompHausLike.of _ X


instance (X : Stonean.{u}) : ExtremallyDisconnected X := X.prop


/-- The functor from Stonean spaces to profinite spaces. -/
abbrev toProfinite : Stonean.{u} ⥤ Profinite.{u} :=
  CompHausLike.toCompHausLike (fun _ ↦ inferInstance)


instance (X : Stonean.{u}) : ExtremallyDisconnected ((forget _).obj X) := X.prop


instance (X : Stonean.{u}) : TotallyDisconnectedSpace ((forget _).obj X) :=
  show TotallyDisconnectedSpace X from inferInstance


/--
A finite discrete space as a Stonean space.
-/
def mkFinite (X : Type*) [Finite X] [TopologicalSpace X] [DiscreteTopology X] : Stonean where
  toTop := (CompHaus.of X).toTop
  prop := by
    /-
      X : Type u_1
      inst✝² : Finite X
      inst✝¹ : TopologicalSpace X
      inst✝ : DiscreteTopology X
      ⊢ ExtremallyDisconnected ↑(CompHaus.of X).toTop
    -/
    dsimp
    /-
      X : Type u_1
      inst✝² : Finite X
      inst✝¹ : TopologicalSpace X
      inst✝ : DiscreteTopology X
      ⊢ ExtremallyDisconnected X
    -/
    constructor
    /-
      case open_closure
      X : Type u_1
      inst✝² : Finite X
      inst✝¹ : TopologicalSpace X
      inst✝ : DiscreteTopology X
      ⊢ ∀ (U : Set X), IsOpen U → IsOpen (closure U)
    -/
    intro U _
    /-
      case open_closure
      X : Type u_1
      inst✝² : Finite X
      inst✝¹ : TopologicalSpace X
      inst✝ : DiscreteTopology X
      U : Set X
      a✝ : IsOpen U
      ⊢ IsOpen (closure U)
    -/
    apply isOpen_discrete (closure U)
    /-
      🎉 no goals
    -/


/--
A morphism in `Stonean` is an epi iff it is surjective.
-/
lemma epi_iff_surjective {X Y : Stonean} (f : X ⟶ Y) :
    Epi f ↔ Function.Surjective f := by
  /-
    X Y : Stonean
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Epi f) (Function.Surjective ⇑f)
  -/
  refine ⟨?_, ConcreteCategory.epi_of_surjective _⟩
  /-
    X Y : Stonean
    f : Quiver.Hom X Y
    ⊢ CategoryTheory.Epi f → Function.Surjective ⇑f
  -/
  dsimp [Function.Surjective]
  /-
    X Y : Stonean
    f : Quiver.Hom X Y
    ⊢ CategoryTheory.Epi f → ∀ (b : (CategoryTheory.forget Stonean).obj Y), Exists …
  -/
  intro h y
  /-
    X Y : Stonean
    f : Quiver.Hom X Y
    h : CategoryTheory.Epi f
    y : (CategoryTheory.forget Stonean).obj Y
    ⊢ Exists fun a => Eq (f a) y
  -/
  by_contra! hy
  /-
    X Y : Stonean
    f : Quiver.Hom X Y
    h : CategoryTheory.Epi f
    y : (CategoryTheory.forget Stonean).obj Y
    hy : ∀ (a : (CategoryTheory.forget Stonean).obj X), Ne (f a) y
    ⊢ False
  -/
  let C := Set.range f
  /-
    X Y : Stonean
    f : Quiver.Hom X Y
    h : CategoryTheory.Epi f
    y : (CategoryTheory.forget Stonean).obj Y
    hy : ∀ (a : (CategoryTheory.forget Stonean).obj X), Ne (f a) y
    C : Set ((CategoryTheory.forget Stonean).obj Y) := Set.range ⇑f
    ⊢ False
  -/
  have hC : IsClosed C := (isCompact_range f.continuous).isClosed
  /-
    X Y : Stonean
    f : Quiver.Hom X Y
    h : CategoryTheory.Epi f
    y : (CategoryTheory.forget Stonean).obj Y
    hy : ∀ (a : (CategoryTheory.forget Stonean).obj X), Ne (f a) y
    C : Set ((CategoryTheory.forget Stonean).obj Y) := Set.range ⇑f
    hC : IsClosed C
    ⊢ False
  -/
  let U := Cᶜ
  have hUy : U ∈ 𝓝 y := by
    simp only [U, C, Set.mem_range, hy, exists_false, not_false_eq_true, hC.compl_mem_nhds]
  /-
    X Y : Stonean
    f : Quiver.Hom X Y
    h : CategoryTheory.Epi f
    y : (CategoryTheory.forget Stonean).obj Y
    hy : ∀ (a : (CategoryTheory.forget Stonean).obj X), Ne (f a) y
    C : Set ((CategoryTheory.forget Stonean).obj Y) := Set.range ⇑f
    hC : IsClosed C
    U : Set ((CategoryTheory.forget Stonean).obj Y) := HasCompl.compl C
    hUy : Membership.mem (nhds y) U
    ⊢ False
  -/
  obtain ⟨V, hV, hyV, hVU⟩ := isTopologicalBasis_isClopen.mem_nhds_iff.mp hUy
  classical
  let g : Y ⟶ mkFinite (ULift (Fin 2)) :=
    ⟨(LocallyConstant.ofIsClopen hV).map ULift.up, LocallyConstant.continuous _⟩
  let h : Y ⟶ mkFinite (ULift (Fin 2)) := ⟨fun _ => ⟨1⟩, continuous_const⟩
  have H : h = g := by
    rw [← cancel_epi f]
    ext x
    apply ULift.ext -- why is `ext` not doing this automatically?
    change 1 = ite _ _ _ -- why is `dsimp` not getting me here?
    rw [if_neg]
    refine mt (hVU ·) ?_ -- what would be an idiomatic tactic for this step?
    simpa only [U, Set.mem_compl_iff, Set.mem_range, not_exists, not_forall, not_not]
      using exists_apply_eq_apply f x
  apply_fun fun e => (e y).down at H
  change 1 = ite _ _ _ at H -- why is `dsimp at H` not getting me here?
  rw [if_pos hyV] at H
  exact one_ne_zero H


/-- Every Stonean space is projective in `CompHaus` -/
instance instProjectiveCompHausCompHaus (X : Stonean) : Projective (toCompHaus.obj X) where
  factors := by
    /-
      X : Stonean
      ⊢ ∀ {E X_1 : CompHaus} (f : Quiver.Hom (Stonean.toCompHaus.obj X) X_1) (e : Qu …
    -/
    intro B C φ f _
    /-
      X : Stonean
      B C : CompHaus
      φ : Quiver.Hom (Stonean.toCompHaus.obj X) C
      f : Quiver.Hom B C
      inst✝ : CategoryTheory.Epi f
      ⊢ Exists fun f' => Eq (CategoryTheory.CategoryStruct.comp f' f) φ
    -/
    haveI : ExtremallyDisconnected (toCompHaus.obj X).toTop := X.prop
    /-
      X : Stonean
      B C : CompHaus
      φ : Quiver.Hom (Stonean.toCompHaus.obj X) C
      f : Quiver.Hom B C
      inst✝ : CategoryTheory.Epi f
      this : ExtremallyDisconnected ↑(Stonean.toCompHaus.obj X).toTop
      ⊢ Exists fun f' => Eq (CategoryTheory.CategoryStruct.comp f' f) φ
    -/
    have hf : Function.Surjective f := by rwa [← CompHaus.epi_iff_surjective]
    /-
      X : Stonean
      B C : CompHaus
      φ : Quiver.Hom (Stonean.toCompHaus.obj X) C
      f : Quiver.Hom B C
      inst✝ : CategoryTheory.Epi f
      this : ExtremallyDisconnected ↑(Stonean.toCompHaus.obj X).toTop
      hf : Function.Surjective ⇑f
      ⊢ Exists fun f' => Eq (CategoryTheory.CategoryStruct.comp f' f) φ
    -/
    obtain ⟨f', h⟩ := CompactT2.ExtremallyDisconnected.projective φ.continuous f.continuous hf
    /-
      case intro
      X : Stonean
      B C : CompHaus
      φ : Quiver.Hom (Stonean.toCompHaus.obj X) C
      f : Quiver.Hom B C
      inst✝ : CategoryTheory.Epi f
      this : ExtremallyDisconnected ↑(Stonean.toCompHaus.obj X).toTop
      hf : Function.Surjective ⇑f
      f' : ↑(Stonean.toCompHaus.obj X).toTop → ↑B.toTop
      h : And (Continuous f') (Eq (Function.comp (⇑f) f') ⇑φ)
      ⊢ Exists fun f' => Eq (CategoryTheory.CategoryStruct.comp f' f) φ
    -/
    use ⟨f', h.left⟩
    /-
      case h
      X : Stonean
      B C : CompHaus
      φ : Quiver.Hom (Stonean.toCompHaus.obj X) C
      f : Quiver.Hom B C
      inst✝ : CategoryTheory.Epi f
      this : ExtremallyDisconnected ↑(Stonean.toCompHaus.obj X).toTop
      hf : Function.Surjective ⇑f
      f' : ↑(Stonean.toCompHaus.obj X).toTop → ↑B.toTop
      h : And (Continuous f') (Eq (Function.comp (⇑f) f') ⇑φ)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := f', continuous_toFun := ⋯  …
    -/
    ext
    /-
      case h.w
      X : Stonean
      B C : CompHaus
      φ : Quiver.Hom (Stonean.toCompHaus.obj X) C
      f : Quiver.Hom B C
      inst✝ : CategoryTheory.Epi f
      this : ExtremallyDisconnected ↑(Stonean.toCompHaus.obj X).toTop
      hf : Function.Surjective ⇑f
      f' : ↑(Stonean.toCompHaus.obj X).toTop → ↑B.toTop
      h : And (Continuous f') (Eq (Function.comp (⇑f) f') ⇑φ)
      x✝ : (CategoryTheory.forget CompHaus).obj (Stonean.toCompHaus.obj X)
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { toFun := f', continuous_toFun := ⋯ …
    -/
    exact congr_fun h.right _
    /-
      🎉 no goals
    -/


/-- Every Stonean space is projective in `Profinite` -/
instance (X : Stonean) : Projective (toProfinite.obj X) where
  factors := by
    /-
      X : Stonean
      ⊢ ∀ {E X_1 : Profinite} (f : Quiver.Hom (Stonean.toProfinite.obj X) X_1) (e :  …
    -/
    intro B C φ f _
    /-
      X : Stonean
      B C : Profinite
      φ : Quiver.Hom (Stonean.toProfinite.obj X) C
      f : Quiver.Hom B C
      inst✝ : CategoryTheory.Epi f
      ⊢ Exists fun f' => Eq (CategoryTheory.CategoryStruct.comp f' f) φ
    -/
    haveI : ExtremallyDisconnected (toProfinite.obj X) := X.prop
    /-
      X : Stonean
      B C : Profinite
      φ : Quiver.Hom (Stonean.toProfinite.obj X) C
      f : Quiver.Hom B C
      inst✝ : CategoryTheory.Epi f
      this : ExtremallyDisconnected ↑(Stonean.toProfinite.obj X).toTop
      ⊢ Exists fun f' => Eq (CategoryTheory.CategoryStruct.comp f' f) φ
    -/
    have hf : Function.Surjective f := by rwa [← Profinite.epi_iff_surjective]
    /-
      X : Stonean
      B C : Profinite
      φ : Quiver.Hom (Stonean.toProfinite.obj X) C
      f : Quiver.Hom B C
      inst✝ : CategoryTheory.Epi f
      this : ExtremallyDisconnected ↑(Stonean.toProfinite.obj X).toTop
      hf : Function.Surjective ⇑f
      ⊢ Exists fun f' => Eq (CategoryTheory.CategoryStruct.comp f' f) φ
    -/
    obtain ⟨f', h⟩ := CompactT2.ExtremallyDisconnected.projective φ.continuous f.continuous hf
    /-
      case intro
      X : Stonean
      B C : Profinite
      φ : Quiver.Hom (Stonean.toProfinite.obj X) C
      f : Quiver.Hom B C
      inst✝ : CategoryTheory.Epi f
      this : ExtremallyDisconnected ↑(Stonean.toProfinite.obj X).toTop
      hf : Function.Surjective ⇑f
      f' : ↑(Stonean.toProfinite.obj X).toTop → ↑B.toTop
      h : And (Continuous f') (Eq (Function.comp (⇑f) f') ⇑φ)
      ⊢ Exists fun f' => Eq (CategoryTheory.CategoryStruct.comp f' f) φ
    -/
    use ⟨f', h.left⟩
    /-
      case h
      X : Stonean
      B C : Profinite
      φ : Quiver.Hom (Stonean.toProfinite.obj X) C
      f : Quiver.Hom B C
      inst✝ : CategoryTheory.Epi f
      this : ExtremallyDisconnected ↑(Stonean.toProfinite.obj X).toTop
      hf : Function.Surjective ⇑f
      f' : ↑(Stonean.toProfinite.obj X).toTop → ↑B.toTop
      h : And (Continuous f') (Eq (Function.comp (⇑f) f') ⇑φ)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := f', continuous_toFun := ⋯  …
    -/
    ext
    /-
      case h.w
      X : Stonean
      B C : Profinite
      φ : Quiver.Hom (Stonean.toProfinite.obj X) C
      f : Quiver.Hom B C
      inst✝ : CategoryTheory.Epi f
      this : ExtremallyDisconnected ↑(Stonean.toProfinite.obj X).toTop
      hf : Function.Surjective ⇑f
      f' : ↑(Stonean.toProfinite.obj X).toTop → ↑B.toTop
      h : And (Continuous f') (Eq (Function.comp (⇑f) f') ⇑φ)
      x✝ : (CategoryTheory.forget Profinite).obj (Stonean.toProfinite.obj X)
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { toFun := f', continuous_toFun := ⋯ …
    -/
    exact congr_fun h.right _
    /-
      🎉 no goals
    -/


/-- Every Stonean space is projective in `Stonean`. -/
instance (X : Stonean) : Projective X where
  factors := by
    /-
      X : Stonean
      ⊢ ∀ {E X_1 : Stonean} (f : Quiver.Hom X X_1) (e : Quiver.Hom E X_1) [inst : Ca …
    -/
    intro B C φ f _
    /-
      X B C : Stonean
      φ : Quiver.Hom X C
      f : Quiver.Hom B C
      inst✝ : CategoryTheory.Epi f
      ⊢ Exists fun f' => Eq (CategoryTheory.CategoryStruct.comp f' f) φ
    -/
    haveI : ExtremallyDisconnected X.toTop := X.prop
    /-
      X B C : Stonean
      φ : Quiver.Hom X C
      f : Quiver.Hom B C
      inst✝ : CategoryTheory.Epi f
      this : ExtremallyDisconnected ↑X.toTop
      ⊢ Exists fun f' => Eq (CategoryTheory.CategoryStruct.comp f' f) φ
    -/
    have hf : Function.Surjective f := by rwa [← Stonean.epi_iff_surjective]
    /-
      X B C : Stonean
      φ : Quiver.Hom X C
      f : Quiver.Hom B C
      inst✝ : CategoryTheory.Epi f
      this : ExtremallyDisconnected ↑X.toTop
      hf : Function.Surjective ⇑f
      ⊢ Exists fun f' => Eq (CategoryTheory.CategoryStruct.comp f' f) φ
    -/
    obtain ⟨f', h⟩ := CompactT2.ExtremallyDisconnected.projective φ.continuous f.continuous hf
    /-
      case intro
      X B C : Stonean
      φ : Quiver.Hom X C
      f : Quiver.Hom B C
      inst✝ : CategoryTheory.Epi f
      this : ExtremallyDisconnected ↑X.toTop
      hf : Function.Surjective ⇑f
      f' : ↑X.toTop → ↑B.toTop
      h : And (Continuous f') (Eq (Function.comp (⇑f) f') ⇑φ)
      ⊢ Exists fun f' => Eq (CategoryTheory.CategoryStruct.comp f' f) φ
    -/
    use ⟨f', h.left⟩
    /-
      case h
      X B C : Stonean
      φ : Quiver.Hom X C
      f : Quiver.Hom B C
      inst✝ : CategoryTheory.Epi f
      this : ExtremallyDisconnected ↑X.toTop
      hf : Function.Surjective ⇑f
      f' : ↑X.toTop → ↑B.toTop
      h : And (Continuous f') (Eq (Function.comp (⇑f) f') ⇑φ)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := f', continuous_toFun := ⋯  …
    -/
    ext
    /-
      case h.w
      X B C : Stonean
      φ : Quiver.Hom X C
      f : Quiver.Hom B C
      inst✝ : CategoryTheory.Epi f
      this : ExtremallyDisconnected ↑X.toTop
      hf : Function.Surjective ⇑f
      f' : ↑X.toTop → ↑B.toTop
      h : And (Continuous f') (Eq (Function.comp (⇑f) f') ⇑φ)
      x✝ : (CategoryTheory.forget Stonean).obj X
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { toFun := f', continuous_toFun := ⋯ …
    -/
    exact congr_fun h.right _
    /-
      🎉 no goals
    -/


/-- If `X` is compact Hausdorff, `presentation X` is a Stonean space equipped with an epimorphism
  down to `X` (see `CompHaus.presentation.π` and `CompHaus.presentation.epi_π`). It is a
  "constructive" witness to the fact that `CompHaus` has enough projectives. -/
noncomputable
def presentation (X : CompHaus) : Stonean where
  toTop := (projectivePresentation X).p.1
  prop := by
    refine CompactT2.Projective.extremallyDisconnected
      (@fun Y Z _ _ _ _ _ _ f g hfcont hgcont hgsurj => ?_)
    /-
      X : CompHaus
      Y Z : Type ?u.15937
      x✝⁵ : TopologicalSpace Y
      x✝⁴ : TopologicalSpace Z
      x✝³ : CompactSpace Y
      x✝² : T2Space Y
      x✝¹ : CompactSpace Z
      x✝ : T2Space Z
      f : ↑X.projectivePresentation.p.toTop → Z
      g : Y → Z
      hfcont : Continuous f
      hgcont : Continuous g
      hgsurj : Function.Surjective g
      ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp g h) f)
    -/
    let g₁ : (CompHaus.of Y) ⟶ (CompHaus.of Z) := ⟨g, hgcont⟩
    /-
      X : CompHaus
      Y Z : Type ?u.16263
      x✝⁵ : TopologicalSpace Y
      x✝⁴ : TopologicalSpace Z
      x✝³ : CompactSpace Y
      x✝² : T2Space Y
      x✝¹ : CompactSpace Z
      x✝ : T2Space Z
      f : ↑X.projectivePresentation.p.toTop → Z
      g : Y → Z
      hfcont : Continuous f
      hgcont : Continuous g
      hgsurj : Function.Surjective g
      g₁ : Quiver.Hom (CompHaus.of Y) (CompHaus.of Z) := { toFun := g, continuous_to …
      ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp g h) f)
    -/
    let f₁ : (projectivePresentation X).p ⟶ (CompHaus.of Z) := ⟨f, hfcont⟩
    /-
      X : CompHaus
      Y Z : Type ?u.16519
      x✝⁵ : TopologicalSpace Y
      x✝⁴ : TopologicalSpace Z
      x✝³ : CompactSpace Y
      x✝² : T2Space Y
      x✝¹ : CompactSpace Z
      x✝ : T2Space Z
      f : ↑X.projectivePresentation.p.toTop → Z
      g : Y → Z
      hfcont : Continuous f
      hgcont : Continuous g
      hgsurj : Function.Surjective g
      g₁ : Quiver.Hom (CompHaus.of Y) (CompHaus.of Z) := { toFun := g, continuous_to …
      f₁ : Quiver.Hom X.projectivePresentation.p (CompHaus.of Z) := { toFun := f, co …
      ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp g h) f)
    -/
    have hg₁ : Epi g₁ := (epi_iff_surjective _).2 hgsurj
    /-
      X : CompHaus
      Y Z : Type ?u.16519
      x✝⁵ : TopologicalSpace Y
      x✝⁴ : TopologicalSpace Z
      x✝³ : CompactSpace Y
      x✝² : T2Space Y
      x✝¹ : CompactSpace Z
      x✝ : T2Space Z
      f : ↑X.projectivePresentation.p.toTop → Z
      g : Y → Z
      hfcont : Continuous f
      hgcont : Continuous g
      hgsurj : Function.Surjective g
      g₁ : Quiver.Hom (CompHaus.of Y) (CompHaus.of Z) := { toFun := g, continuous_to …
      f₁ : Quiver.Hom X.projectivePresentation.p (CompHaus.of Z) := { toFun := f, co …
      hg₁ : CategoryTheory.Epi g₁
      ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp g h) f)
    -/
    refine ⟨Projective.factorThru f₁ g₁, (Projective.factorThru f₁ g₁).2, funext (fun _ => ?_)⟩
    /-
      X : CompHaus
      Y Z : Type ?u.16519
      x✝⁶ : TopologicalSpace Y
      x✝⁵ : TopologicalSpace Z
      x✝⁴ : CompactSpace Y
      x✝³ : T2Space Y
      x✝² : CompactSpace Z
      x✝¹ : T2Space Z
      f : ↑X.projectivePresentation.p.toTop → Z
      g : Y → Z
      hfcont : Continuous f
      hgcont : Continuous g
      hgsurj : Function.Surjective g
      g₁ : Quiver.Hom (CompHaus.of Y) (CompHaus.of Z) := { toFun := g, continuous_to …
      f₁ : Quiver.Hom X.projectivePresentation.p (CompHaus.of Z) := { toFun := f, co …
      hg₁ : CategoryTheory.Epi g₁
      x✝ : ↑X.projectivePresentation.p.toTop
      ⊢ Eq (Function.comp g (⇑(CategoryTheory.Projective.factorThru f₁ g₁)) x✝) (f x✝)
    -/
    change (Projective.factorThru f₁ g₁ ≫ g₁) _ = f _
    /-
      X : CompHaus
      Y Z : Type ?u.16519
      x✝⁶ : TopologicalSpace Y
      x✝⁵ : TopologicalSpace Z
      x✝⁴ : CompactSpace Y
      x✝³ : T2Space Y
      x✝² : CompactSpace Z
      x✝¹ : T2Space Z
      f : ↑X.projectivePresentation.p.toTop → Z
      g : Y → Z
      hfcont : Continuous f
      hgcont : Continuous g
      hgsurj : Function.Surjective g
      g₁ : Quiver.Hom (CompHaus.of Y) (CompHaus.of Z) := { toFun := g, continuous_to …
      f₁ : Quiver.Hom X.projectivePresentation.p (CompHaus.of Z) := { toFun := f, co …
      hg₁ : CategoryTheory.Epi g₁
      x✝ : ↑X.projectivePresentation.p.toTop
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Projective.factorThr …
    -/
    rw [Projective.factorThru_comp]
    /-
      X : CompHaus
      Y Z : Type ?u.16519
      x✝⁶ : TopologicalSpace Y
      x✝⁵ : TopologicalSpace Z
      x✝⁴ : CompactSpace Y
      x✝³ : T2Space Y
      x✝² : CompactSpace Z
      x✝¹ : T2Space Z
      f : ↑X.projectivePresentation.p.toTop → Z
      g : Y → Z
      hfcont : Continuous f
      hgcont : Continuous g
      hgsurj : Function.Surjective g
      g₁ : Quiver.Hom (CompHaus.of Y) (CompHaus.of Z) := { toFun := g, continuous_to …
      f₁ : Quiver.Hom X.projectivePresentation.p (CompHaus.of Z) := { toFun := f, co …
      hg₁ : CategoryTheory.Epi g₁
      x✝ : ↑X.projectivePresentation.p.toTop
      ⊢ Eq (f₁ x✝) (f x✝)
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The morphism from `presentation X` to `X`. -/
noncomputable
def presentation.π (X : CompHaus) : Stonean.toCompHaus.obj X.presentation ⟶ X :=
  (projectivePresentation X).f


/-- The morphism from `presentation X` to `X` is an epimorphism. -/
noncomputable
instance presentation.epi_π (X : CompHaus) : Epi (π X) :=
  (projectivePresentation X).epi


/-- The underlying `CompHaus` of a `Stonean`. -/
abbrev _root_.Stonean.compHaus (X : Stonean) := Stonean.toCompHaus.obj X


/--
```
               X
               |
              (f)
               |
               \/
  Z ---(e)---> Y
```
If `Z` is a Stonean space, `f : X ⟶ Y` an epi in `CompHaus` and `e : Z ⟶ Y` is arbitrary, then
`lift e f` is a fixed (but arbitrary) lift of `e` to a morphism `Z ⟶ X`. It exists because
`Z` is a projective object in `CompHaus`.
-/
noncomputable
def lift {X Y : CompHaus} {Z : Stonean} (e : Z.compHaus ⟶ Y) (f : X ⟶ Y) [Epi f] :
    Z.compHaus ⟶ X :=
  Projective.factorThru e f


@[simp, reassoc]
lemma lift_lifts {X Y : CompHaus} {Z : Stonean} (e : Z.compHaus ⟶ Y) (f : X ⟶ Y) [Epi f] :
                           /-
                             X Y : CompHaus
                             Z : Stonean
                             e : Quiver.Hom Z.compHaus Y
                             f : Quiver.Hom X Y
                             inst✝ : CategoryTheory.Epi f
                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (CompHaus.lift e f) f) e
                           -/
    lift e f ≫ f = e := by simp [lift]
                           /-
                             🎉 no goals
                           -/


lemma Gleason (X : CompHaus.{u}) :
    Projective X ↔ ExtremallyDisconnected X := by
  /-
    X : CompHaus
    ⊢ Iff (CategoryTheory.Projective X) (ExtremallyDisconnected ↑X.toTop)
  -/
  constructor
    /-
      case mp
      X : CompHaus
      ⊢ CategoryTheory.Projective X → ExtremallyDisconnected ↑X.toTop
    -/
  · intro h
    /-
      case mp
      X : CompHaus
      h : CategoryTheory.Projective X
      ⊢ ExtremallyDisconnected ↑X.toTop
    -/
    show ExtremallyDisconnected X.toStonean
    /-
      case mp
      X : CompHaus
      h : CategoryTheory.Projective X
      ⊢ ExtremallyDisconnected ↑X.toStonean.toTop
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : CompHaus
      ⊢ ExtremallyDisconnected ↑X.toTop → CategoryTheory.Projective X
    -/
  · intro h
    /-
      case mpr
      X : CompHaus
      h : ExtremallyDisconnected ↑X.toTop
      ⊢ CategoryTheory.Projective X
    -/
    let X' : Stonean := ⟨X.toTop, inferInstance⟩
    /-
      case mpr
      X : CompHaus
      h : ExtremallyDisconnected ↑X.toTop
      X' : Stonean := CompHausLike.mk X.toTop ⋯
      ⊢ CategoryTheory.Projective X
    -/
    show Projective X'.compHaus
    /-
      case mpr
      X : CompHaus
      h : ExtremallyDisconnected ↑X.toTop
      X' : Stonean := CompHausLike.mk X.toTop ⋯
      ⊢ CategoryTheory.Projective X'.compHaus
    -/
    apply Stonean.instProjectiveCompHausCompHaus
    /-
      🎉 no goals
    -/


/-- If `X` is profinite, `presentation X` is a Stonean space equipped with an epimorphism down to
    `X` (see `Profinite.presentation.π` and `Profinite.presentation.epi_π`). -/
noncomputable
def presentation (X : Profinite) : Stonean where
  toTop := (profiniteToCompHaus.obj X).projectivePresentation.p.toTop
  prop := (profiniteToCompHaus.obj X).presentation.prop


/-- The morphism from `presentation X` to `X`. -/
noncomputable
def presentation.π (X : Profinite) : Stonean.toProfinite.obj X.presentation ⟶ X :=
  (profiniteToCompHaus.obj X).projectivePresentation.f


/-- The morphism from `presentation X` to `X` is an epimorphism. -/
noncomputable
instance presentation.epi_π (X : Profinite) : Epi (π X) := by
  /-
    X : Profinite
    ⊢ CategoryTheory.Epi (Profinite.presentation.π X)
  -/
  have := (profiniteToCompHaus.obj X).projectivePresentation.epi
  /-
    X : Profinite
    this : CategoryTheory.Epi (profiniteToCompHaus.obj X).projectivePresentation.f
    ⊢ CategoryTheory.Epi (Profinite.presentation.π X)
  -/
  rw [CompHaus.epi_iff_surjective] at this
  /-
    X : Profinite
    this : Function.Surjective ⇑(profiniteToCompHaus.obj X).projectivePresentation.f
    ⊢ CategoryTheory.Epi (Profinite.presentation.π X)
  -/
  rw [epi_iff_surjective]
  /-
    X : Profinite
    this : Function.Surjective ⇑(profiniteToCompHaus.obj X).projectivePresentation.f
    ⊢ Function.Surjective ⇑(Profinite.presentation.π X)
  -/
  exact this
  /-
    🎉 no goals
  -/


/--
```
               X
               |
              (f)
               |
               \/
  Z ---(e)---> Y
```
If `Z` is a Stonean space, `f : X ⟶ Y` an epi in `Profinite` and `e : Z ⟶ Y` is arbitrary,
then `lift e f` is a fixed (but arbitrary) lift of `e` to a morphism `Z ⟶ X`. It is
`CompHaus.lift e f` as a morphism in `Profinite`.
-/
noncomputable
def lift {X Y : Profinite} {Z : Stonean} (e : Stonean.toProfinite.obj Z ⟶ Y) (f : X ⟶ Y) [Epi f] :
    Stonean.toProfinite.obj Z ⟶ X := Projective.factorThru e f


@[simp, reassoc]
lemma lift_lifts {X Y : Profinite} {Z : Stonean} (e : Stonean.toProfinite.obj Z ⟶ Y) (f : X ⟶ Y)
                                     /-
                                       X Y : Profinite
                                       Z : Stonean
                                       e : Quiver.Hom (Stonean.toProfinite.obj Z) Y
                                       f : Quiver.Hom X Y
                                       inst✝ : CategoryTheory.Epi f
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (Profinite.lift e f) f) e
                                     -/
    [Epi f] : lift e f ≫ f = e := by simp [lift]
                                     /-
                                       🎉 no goals
                                     -/


lemma projective_of_extrDisc {X : Profinite.{u}} (hX : ExtremallyDisconnected X) :
    Projective X := by
  /-
    X : Profinite
    hX : ExtremallyDisconnected ↑X.toTop
    ⊢ CategoryTheory.Projective X
  -/
  show Projective (Stonean.toProfinite.obj ⟨X.toTop, inferInstance⟩)
  /-
    X : Profinite
    hX : ExtremallyDisconnected ↑X.toTop
    ⊢ CategoryTheory.Projective (Stonean.toProfinite.obj (CompHausLike.mk X.toTop  …
  -/
  exact inferInstance
  /-
    🎉 no goals
  -/


