/--
The equivalence between `K.sections` and `(K ⋙ uliftFunctor.{v, u}).sections`. This is used to show
that `uliftFunctor` preserves limits that are potentially too large to exist in the source
category.
-/
def sectionsEquiv {J : Type*} [Category J] (K : J ⥤ Type u) :
    K.sections ≃ (K ⋙ uliftFunctor.{v, u}).sections where
                                                       /-
                                                         J : Type u_1
                                                         inst✝ : CategoryTheory.Category.{?u.6, u_1} J
                                                         K : CategoryTheory.Functor J (Type u)
                                                         x✝ : ↑K.sections
                                                         u : (j : J) → K.obj j
                                                         hu : Membership.mem K.sections u
                                                         j✝ j'✝ : J
                                                         f : Quiver.Hom j✝ j'✝
                                                         ⊢ Eq ((K.comp CategoryTheory.uliftFunctor.{v, u}).map f ((fun j => { down := u …
                                                       -/
  toFun := fun ⟨u, hu⟩ => ⟨fun j => ⟨u j⟩, fun f => by simp [hu f]⟩
                                                       /-
                                                         🎉 no goals
                                                       -/
                                                                   /-
                                                                     J : Type u_1
                                                                     inst✝ : CategoryTheory.Category.{?u.6, u_1} J
                                                                     K : CategoryTheory.Functor J (Type u)
                                                                     x✝ : ↑(K.comp CategoryTheory.uliftFunctor.{v, u}).sections
                                                                     u : (j : J) → (K.comp CategoryTheory.uliftFunctor.{v, u}).obj j
                                                                     hu : Membership.mem (K.comp CategoryTheory.uliftFunctor.{v, u}).sections u
                                                                     j j' : J
                                                                     f : Quiver.Hom j j'
                                                                     ⊢ Eq (K.map f ((fun j => (u j).down) j)) ((fun j => (u j).down) j')
                                                                   -/
  invFun := fun ⟨u, hu⟩ => ⟨fun j => (u j).down, @fun j j' f => by simp [← hu f]⟩
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  left_inv _ := rfl
  right_inv _ := rfl


/--
The functor `uliftFunctor : Type u ⥤ Type (max u v)` preserves limits of arbitrary size.
-/
noncomputable
instance : PreservesLimitsOfSize.{w', w} uliftFunctor.{v, u} where
  preservesLimitsOfShape {J} := {
    preservesLimit := fun {K} => {
      preserves := fun {c} hc => by
        /-
          J : Type w
          inst✝ : CategoryTheory.Category.{w', w} J
          K : CategoryTheory.Functor J (Type u)
          c : CategoryTheory.Limits.Cone K
          hc : CategoryTheory.Limits.IsLimit c
          ⊢ Nonempty (CategoryTheory.Limits.IsLimit (CategoryTheory.uliftFunctor.{v, u}. …
        -/
        rw [Types.isLimit_iff ((uliftFunctor.{v, u}).mapCone c)]
        /-
          J : Type w
          inst✝ : CategoryTheory.Category.{w', w} J
          K : CategoryTheory.Functor J (Type u)
          c : CategoryTheory.Limits.Cone K
          hc : CategoryTheory.Limits.IsLimit c
          ⊢ ∀ (s : (j : J) → (K.comp CategoryTheory.uliftFunctor.{v, u}).obj j), Members …
        -/
        intro s hs
        /-
          J : Type w
          inst✝ : CategoryTheory.Category.{w', w} J
          K : CategoryTheory.Functor J (Type u)
          c : CategoryTheory.Limits.Cone K
          hc : CategoryTheory.Limits.IsLimit c
          s : (j : J) → (K.comp CategoryTheory.uliftFunctor.{v, u}).obj j
          hs : Membership.mem (K.comp CategoryTheory.uliftFunctor.{v, u}).sections s
          ⊢ ExistsUnique fun x => ∀ (j : J), Eq ((CategoryTheory.uliftFunctor.{v, u}.map …
        -/
        obtain ⟨x, hx₁, hx₂⟩ := (Types.isLimit_iff c).mp ⟨hc⟩ _ ((sectionsEquiv K).symm ⟨s, hs⟩).2
        exact ⟨⟨x⟩, fun i => ULift.ext _ _ (hx₁ i),
          fun y hy => ULift.ext _ _ (hx₂ y.down fun i ↦ ULift.ext_iff.mp (hy i))⟩ } }


/--
The functor `uliftFunctor : Type u ⥤ Type (max u v)` creates `u`-small limits.
-/
noncomputable instance : CreatesLimitsOfSize.{w, u} uliftFunctor.{v, u} where
  CreatesLimitsOfShape := { CreatesLimit := fun {_} ↦ createsLimitOfFullyFaithfulOfPreserves }


/-- Given a subset of the cocone point of a cocone over the lifted functor,
  produce a cocone over the original functor. -/
def coconeOfSet (ls : Set lc.pt) : Cocone K where
  pt := ULift Prop
  ι :=
  { app := fun j x ↦ ⟨lc.ι.app j ⟨x⟩ ∈ ls⟩
                                 /-
                                   J : Type u_1
                                   inst✝ : CategoryTheory.Category.{?u.6341, u_1} J
                                   K : CategoryTheory.Functor J (Type u)
                                   c : CategoryTheory.Limits.Cocone K
                                   hc : CategoryTheory.Limits.IsColimit c
                                   lc : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
                                   ls : Set lc.pt
                                   i j : J
                                   f : Quiver.Hom i j
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.map f) ((fun j x => { down := Memb …
                                 -/
    naturality := fun i j f ↦ by dsimp only; rw [← lc.w f]; rfl }
                                                            /-
                                                              🎉 no goals
                                                            -/


/-- Given a subset of the cocone point of a cocone over the lifted functor,
  produce a subset of the cocone point of a colimit cocone over the original functor. -/
def descSet (ls : Set lc.pt) : Set c.pt := {x | (hc.desc (coconeOfSet ls) x).down}


/-- Characterization the map `descSet hc`: the image of an element in a vertex of the original
  diagram in the cocone point lies in `descSet hc ls` if and only if the image of the corresponding
  element in the lifted diagram lie in `ls`. -/
lemma descSet_spec (s : Set c.pt) (ls : Set lc.pt) :
    descSet hc ls = s ↔ ∀ j x, lc.ι.app j ⟨x⟩ ∈ ls ↔ c.ι.app j x ∈ s := by
  /-
    J : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} J
    K : CategoryTheory.Functor J (Type u)
    c : CategoryTheory.Limits.Cocone K
    hc : CategoryTheory.Limits.IsColimit c
    lc : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
    s : Set c.pt
    ls : Set lc.pt
    ⊢ Iff (Eq (CategoryTheory.Limits.Types.descSet hc ls) s) (∀ (j : J) (x : K.obj …
  -/
  refine ⟨?_, fun he ↦ funext fun x ↦ ?_⟩
    /-
      case refine_1
      J : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} J
      K : CategoryTheory.Functor J (Type u)
      c : CategoryTheory.Limits.Cocone K
      hc : CategoryTheory.Limits.IsColimit c
      lc : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
      s : Set c.pt
      ls : Set lc.pt
      ⊢ Eq (CategoryTheory.Limits.Types.descSet hc ls) s → ∀ (j : J) (x : K.obj j),  …
    -/
  · rintro rfl j x
    /-
      case refine_1
      J : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} J
      K : CategoryTheory.Functor J (Type u)
      c : CategoryTheory.Limits.Cocone K
      hc : CategoryTheory.Limits.IsColimit c
      lc : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
      ls : Set lc.pt
      j : J
      x : K.obj j
      ⊢ Iff (Membership.mem ls (lc.ι.app j { down := x })) (Membership.mem (Category …
    -/
    exact (congr_arg ULift.down (congr_fun (hc.fac (coconeOfSet ls) j) x).symm).to_iff
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      J : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} J
      K : CategoryTheory.Functor J (Type u)
      c : CategoryTheory.Limits.Cocone K
      hc : CategoryTheory.Limits.IsColimit c
      lc : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
      s : Set c.pt
      ls : Set lc.pt
      he : ∀ (j : J) (x : K.obj j), Iff (Membership.mem ls (lc.ι.app j { down := x } …
      x : c.pt
      ⊢ Eq (CategoryTheory.Limits.Types.descSet hc ls x) (s x)
    -/
  · refine (congr_arg ULift.down (congr_fun (hc.uniq (coconeOfSet ls) (⟨· ∈ s⟩) fun j ↦ ?_) x)).symm
    /-
      case refine_2
      J : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} J
      K : CategoryTheory.Functor J (Type u)
      c : CategoryTheory.Limits.Cocone K
      hc : CategoryTheory.Limits.IsColimit c
      lc : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
      s : Set c.pt
      ls : Set lc.pt
      he : ∀ (j : J) (x : K.obj j), Iff (Membership.mem ls (lc.ι.app j { down := x } …
      x : c.pt
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) fun x => { down := Member …
    -/
    ext y; exact congr_arg ULift.up (propext (he j y).symm)
           /-
             🎉 no goals
           -/


lemma mem_descSet_singleton {x : lc.pt} {j : J} {y : K.obj j} :
    c.ι.app j y ∈ descSet hc {x} ↔ lc.ι.app j ⟨y⟩ = x :=
  ((descSet_spec hc _ {x}).mp rfl j y).symm


                                                                   /-
                                                                     J : Type u_1
                                                                     inst✝ : CategoryTheory.Category.{u_2, u_1} J
                                                                     K : CategoryTheory.Functor J (Type u)
                                                                     c : CategoryTheory.Limits.Cocone K
                                                                     hc : CategoryTheory.Limits.IsColimit c
                                                                     lc : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
                                                                     ⊢ Eq (CategoryTheory.Limits.Types.descSet hc Set.univ) Set.univ
                                                                   -/
lemma descSet_univ : descSet hc (@Set.univ lc.pt) = Set.univ := by simp [descSet_spec]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


lemma iUnion_descSet_singleton : ⋃ x : lc.pt, descSet hc {x} = Set.univ := by
  /-
    J : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} J
    K : CategoryTheory.Functor J (Type u)
    c : CategoryTheory.Limits.Cocone K
    hc : CategoryTheory.Limits.IsColimit c
    lc : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
    ⊢ Eq (Set.iUnion fun x => CategoryTheory.Limits.Types.descSet hc (Singleton.si …
  -/
  rw [← descSet_univ hc lc, eq_comm, descSet_spec]
  /-
    J : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} J
    K : CategoryTheory.Functor J (Type u)
    c : CategoryTheory.Limits.Cocone K
    hc : CategoryTheory.Limits.IsColimit c
    lc : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
    ⊢ ∀ (j : J) (x : K.obj j), Iff (Membership.mem Set.univ (lc.ι.app j { down :=  …
  -/
  intro j x
  /-
    J : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} J
    K : CategoryTheory.Functor J (Type u)
    c : CategoryTheory.Limits.Cocone K
    hc : CategoryTheory.Limits.IsColimit c
    lc : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
    j : J
    x : K.obj j
    ⊢ Iff (Membership.mem Set.univ (lc.ι.app j { down := x })) (Membership.mem (Se …
  -/
  erw [true_iff, Set.mem_iUnion]
  /-
    J : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} J
    K : CategoryTheory.Functor J (Type u)
    c : CategoryTheory.Limits.Cocone K
    hc : CategoryTheory.Limits.IsColimit c
    lc : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
    j : J
    x : K.obj j
    ⊢ Exists fun i => Membership.mem (CategoryTheory.Limits.Types.descSet hc (Sing …
  -/
  use lc.ι.app j ⟨x⟩
  /-
    case h
    J : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} J
    K : CategoryTheory.Functor J (Type u)
    c : CategoryTheory.Limits.Cocone K
    hc : CategoryTheory.Limits.IsColimit c
    lc : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
    j : J
    x : K.obj j
    ⊢ Membership.mem (CategoryTheory.Limits.Types.descSet hc (Singleton.singleton  …
  -/
  rw [mem_descSet_singleton]
  /-
    🎉 no goals
  -/


                                                           /-
                                                             J : Type u_1
                                                             inst✝ : CategoryTheory.Category.{u_2, u_1} J
                                                             K : CategoryTheory.Functor J (Type u)
                                                             c : CategoryTheory.Limits.Cocone K
                                                             hc : CategoryTheory.Limits.IsColimit c
                                                             lc : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
                                                             ⊢ Eq (CategoryTheory.Limits.Types.descSet hc EmptyCollection.emptyCollection)  …
                                                           -/
lemma descSet_empty : descSet hc (∅ : Set lc.pt) = ∅ := by simp [descSet_spec]
                                                           /-
                                                             🎉 no goals
                                                           -/


lemma descSet_inter_of_ne (x y : lc.pt) (hn : x ≠ y) : descSet hc {x} ∩ descSet hc {y} = ∅ := by
  /-
    J : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} J
    K : CategoryTheory.Functor J (Type u)
    c : CategoryTheory.Limits.Cocone K
    hc : CategoryTheory.Limits.IsColimit c
    lc : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
    x y : lc.pt
    hn : Ne x y
    ⊢ Eq (Inter.inter (CategoryTheory.Limits.Types.descSet hc (Singleton.singleton …
  -/
  rw [eq_comm, ← descSet_empty hc lc, descSet_spec]
  /-
    J : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} J
    K : CategoryTheory.Functor J (Type u)
    c : CategoryTheory.Limits.Cocone K
    hc : CategoryTheory.Limits.IsColimit c
    lc : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
    x y : lc.pt
    hn : Ne x y
    ⊢ ∀ (j : J) (x_1 : K.obj j), Iff (Membership.mem EmptyCollection.emptyCollecti …
  -/
  intro j z
  /-
    J : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} J
    K : CategoryTheory.Functor J (Type u)
    c : CategoryTheory.Limits.Cocone K
    hc : CategoryTheory.Limits.IsColimit c
    lc : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
    x y : lc.pt
    hn : Ne x y
    j : J
    z : K.obj j
    ⊢ Iff (Membership.mem EmptyCollection.emptyCollection (lc.ι.app j { down := z  …
  -/
  erw [false_iff]
  /-
    J : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} J
    K : CategoryTheory.Functor J (Type u)
    c : CategoryTheory.Limits.Cocone K
    hc : CategoryTheory.Limits.IsColimit c
    lc : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
    x y : lc.pt
    hn : Ne x y
    j : J
    z : K.obj j
    ⊢ Not (Membership.mem (Inter.inter (CategoryTheory.Limits.Types.descSet hc (Si …
  -/
  rintro ⟨hx, hy⟩
  /-
    case intro
    J : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} J
    K : CategoryTheory.Functor J (Type u)
    c : CategoryTheory.Limits.Cocone K
    hc : CategoryTheory.Limits.IsColimit c
    lc : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
    x y : lc.pt
    hn : Ne x y
    j : J
    z : K.obj j
    hx : Membership.mem (CategoryTheory.Limits.Types.descSet hc (Singleton.singlet …
    hy : Membership.mem (CategoryTheory.Limits.Types.descSet hc (Singleton.singlet …
    ⊢ False
  -/
  rw [mem_descSet_singleton] at hx hy
  /-
    case intro
    J : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} J
    K : CategoryTheory.Functor J (Type u)
    c : CategoryTheory.Limits.Cocone K
    hc : CategoryTheory.Limits.IsColimit c
    lc : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
    x y : lc.pt
    hn : Ne x y
    j : J
    z : K.obj j
    hx : Eq (lc.ι.app j { down := z }) x
    hy : Eq (lc.ι.app j { down := z }) y
    ⊢ False
  -/
  exact hn (hx ▸ hy)
  /-
    🎉 no goals
  -/


lemma existsUnique_mem_descSet (x : c.pt) : ∃! y : lc.pt, x ∈ descSet hc {y} :=
  existsUnique_of_exists_of_unique
    (Set.mem_iUnion.mp <| Set.eq_univ_iff_forall.mp (iUnion_descSet_singleton hc lc) x)
    fun y₁ y₂ h₁ h₂ ↦ by_contra fun hn ↦
      Set.eq_empty_iff_forall_not_mem.1 (descSet_inter_of_ne hc lc y₁ y₂ hn) x ⟨h₁, h₂⟩


@[deprecated (since := "2024-12-17")] alias exists_unique_mem_descSet := existsUnique_mem_descSet


/-- Given a colimit cocone in `Type u` and an arbitrary cocone over the diagram lifted to
  `Type (max u v)`, produce a function from the cocone point of the colimit cocone to the
  cocone point of the other cocone, that witnesses the colimit cocone also being a colimit
  in the higher universe. -/
noncomputable def descFun (x : c.pt) : lc.pt := (existsUnique_mem_descSet hc lc x).exists.choose


lemma descFun_apply_spec {x : c.pt} {y : lc.pt} : descFun hc lc x = y ↔ x ∈ descSet hc {y} :=
  have hu := existsUnique_mem_descSet hc lc x
  have hm := hu.exists.choose_spec
  ⟨fun he ↦ he ▸ hm, hu.unique hm⟩


lemma descFun_spec (f : c.pt → lc.pt) :
    f = descFun hc lc ↔ ∀ j, f ∘ c.ι.app j = lc.ι.app j ∘ ULift.up := by
  /-
    J : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} J
    K : CategoryTheory.Functor J (Type u)
    c : CategoryTheory.Limits.Cocone K
    hc : CategoryTheory.Limits.IsColimit c
    lc : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
    f : c.pt → lc.pt
    ⊢ Iff (Eq f (CategoryTheory.Limits.Types.descFun hc lc)) (∀ (j : J), Eq (Funct …
  -/
  refine ⟨?_, fun he ↦ funext fun x ↦ ((descFun_apply_spec hc lc).mpr ?_).symm⟩
    /-
      case refine_1
      J : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} J
      K : CategoryTheory.Functor J (Type u)
      c : CategoryTheory.Limits.Cocone K
      hc : CategoryTheory.Limits.IsColimit c
      lc : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
      f : c.pt → lc.pt
      ⊢ Eq f (CategoryTheory.Limits.Types.descFun hc lc) → ∀ (j : J), Eq (Function.c …
    -/
  · rintro rfl j; ext
    /-
      case refine_1.h
      J : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} J
      K : CategoryTheory.Functor J (Type u)
      c : CategoryTheory.Limits.Cocone K
      hc : CategoryTheory.Limits.IsColimit c
      lc : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
      j : J
      x✝ : K.obj j
      ⊢ Eq (Function.comp (CategoryTheory.Limits.Types.descFun hc lc) (c.ι.app j) x✝ …
    -/
    apply (descFun_apply_spec hc lc).mpr
    /-
      case refine_1.h
      J : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} J
      K : CategoryTheory.Functor J (Type u)
      c : CategoryTheory.Limits.Cocone K
      hc : CategoryTheory.Limits.IsColimit c
      lc : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
      j : J
      x✝ : K.obj j
      ⊢ Membership.mem (CategoryTheory.Limits.Types.descSet hc (Singleton.singleton  …
    -/
    rw [mem_descSet_singleton]; rfl
                                /-
                                  🎉 no goals
                                -/
    /-
      case refine_2
      J : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} J
      K : CategoryTheory.Functor J (Type u)
      c : CategoryTheory.Limits.Cocone K
      hc : CategoryTheory.Limits.IsColimit c
      lc : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
      f : c.pt → lc.pt
      he : ∀ (j : J), Eq (Function.comp f (c.ι.app j)) (Function.comp (lc.ι.app j) U …
      x : c.pt
      ⊢ Membership.mem (CategoryTheory.Limits.Types.descSet hc (Singleton.singleton  …
    -/
  · rw [← (jointly_surjective_of_isColimit hc x).choose_spec.choose_spec, mem_descSet_singleton]
    /-
      case refine_2
      J : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} J
      K : CategoryTheory.Functor J (Type u)
      c : CategoryTheory.Limits.Cocone K
      hc : CategoryTheory.Limits.IsColimit c
      lc : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
      f : c.pt → lc.pt
      he : ∀ (j : J), Eq (Function.comp f (c.ι.app j)) (Function.comp (lc.ι.app j) U …
      x : c.pt
      ⊢ Eq (lc.ι.app ⋯.choose { down := ⋯.choose }) (f (c.ι.app ⋯.choose ⋯.choose))
    -/
    exact (congr_fun (he _) _).symm
    /-
      🎉 no goals
    -/


/--
The functor `uliftFunctor : Type u ⥤ Type (max u v)` preserves colimits of arbitrary size.
-/
noncomputable instance : PreservesColimitsOfSize.{w', w} uliftFunctor.{v, u} where
  preservesColimitsOfShape {J _} :=
  { preservesColimit := fun {F} ↦
    { preserves := fun {c} hc ↦ ⟨{
        desc := fun lc x ↦ descFun hc lc x.down
                             /-
                               J✝ : Type u_1
                               inst✝ : CategoryTheory.Category.{?u.28739, u_1} J✝
                               K : CategoryTheory.Functor J✝ (Type u)
                               c✝ : CategoryTheory.Limits.Cocone K
                               hc✝ : CategoryTheory.Limits.IsColimit c✝
                               lc✝ : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
                               J : Type w
                               x✝ : CategoryTheory.Category.{w', w} J
                               F : CategoryTheory.Functor J (Type u)
                               c : CategoryTheory.Limits.Cocone F
                               hc : CategoryTheory.Limits.IsColimit c
                               lc : CategoryTheory.Limits.Cocone (F.comp CategoryTheory.uliftFunctor.{v, u})
                               j : J
                               ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.uliftFunctor.{v, u}. …
                             -/
        fac := fun lc j ↦ by ext ⟨⟩; apply congr_fun ((descFun_spec hc lc _).mp rfl j)
                                     /-
                                       🎉 no goals
                                     -/
                                 /-
                                   J✝ : Type u_1
                                   inst✝ : CategoryTheory.Category.{?u.28739, u_1} J✝
                                   K : CategoryTheory.Functor J✝ (Type u)
                                   c✝ : CategoryTheory.Limits.Cocone K
                                   hc✝ : CategoryTheory.Limits.IsColimit c✝
                                   lc✝ : CategoryTheory.Limits.Cocone (K.comp CategoryTheory.uliftFunctor.{v, u})
                                   J : Type w
                                   x✝ : CategoryTheory.Category.{w', w} J
                                   F : CategoryTheory.Functor J (Type u)
                                   c : CategoryTheory.Limits.Cocone F
                                   hc : CategoryTheory.Limits.IsColimit c
                                   lc : CategoryTheory.Limits.Cocone (F.comp CategoryTheory.uliftFunctor.{v, u})
                                   f : Quiver.Hom (CategoryTheory.uliftFunctor.{v, u}.mapCocone c).pt lc.pt
                                   hf : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.uliftF …
                                   ⊢ Eq f ((fun lc x => CategoryTheory.Limits.Types.descFun hc lc x.down) lc)
                                 -/
        uniq := fun lc f hf ↦ by ext ⟨⟩; apply congr_fun ((descFun_spec hc lc (f ∘ ULift.up)).mpr
          fun j ↦ funext fun y ↦ congr_fun (hf j) ⟨y⟩) }⟩ } }


/--
The functor `uliftFunctor : Type u ⥤ Type (max u v)` creates `u`-small colimits.
-/
noncomputable instance : CreatesColimitsOfSize.{w, u} uliftFunctor.{v, u} where
  CreatesColimitsOfShape := { CreatesColimit := fun {_} ↦ createsColimitOfFullyFaithfulOfPreserves }


