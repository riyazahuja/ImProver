@[reassoc (attr := simp)]
theorem app_idem : P.p.app X ≫ P.p.app X = P.p.app X :=
  congr_app P.idem X


@[reassoc (attr := simp)]
theorem app_p_comp : P.p.app X ≫ f.f.app X = f.f.app X :=
  congr_app (p_comp f) X


@[reassoc (attr := simp)]
theorem app_comp_p : f.f.app X ≫ Q.p.app X = f.f.app X :=
  congr_app (comp_p f) X


@[reassoc]
theorem app_p_comm : P.p.app X ≫ f.f.app X = f.f.app X ≫ Q.p.app X :=
  congr_app (p_comm f) X


instance functor_category_isIdempotentComplete [IsIdempotentComplete C] :
    IsIdempotentComplete (J ⥤ C) := by
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} J
    inst✝¹ : CategoryTheory.Category.{u_3, u_2} C
    P Q : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
    f : Quiver.Hom P Q
    X : J
    inst✝ : CategoryTheory.IsIdempotentComplete C
    ⊢ CategoryTheory.IsIdempotentComplete (CategoryTheory.Functor J C)
  -/
  refine ⟨fun F p hp => ?_⟩
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} J
    inst✝¹ : CategoryTheory.Category.{u_3, u_2} C
    P Q : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
    f : Quiver.Hom P Q
    X : J
    inst✝ : CategoryTheory.IsIdempotentComplete C
    F : CategoryTheory.Functor J C
    p : Quiver.Hom F F
    hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
    ⊢ Exists fun Y => Exists fun i => Exists fun e => And (Eq (CategoryTheory.Cate …
  -/
  have hC := (isIdempotentComplete_iff_hasEqualizer_of_id_and_idempotent C).mp inferInstance
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} J
    inst✝¹ : CategoryTheory.Category.{u_3, u_2} C
    P Q : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
    f : Quiver.Hom P Q
    X : J
    inst✝ : CategoryTheory.IsIdempotentComplete C
    F : CategoryTheory.Functor J C
    p : Quiver.Hom F F
    hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
    hC : ∀ (X : C) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp p  …
    ⊢ Exists fun Y => Exists fun i => Exists fun e => And (Eq (CategoryTheory.Cate …
  -/
  haveI : ∀ j : J, HasEqualizer (𝟙 _) (p.app j) := fun j => hC _ _ (congr_app hp j)
  /- We construct the direct factor `Y` associated to `p : F ⟶ F` by computing
      the equalizer of the identity and `p.app j` on each object `(j : J)`. -/
  let Y : J ⥤ C :=
    { obj := fun j => Limits.equalizer (𝟙 _) (p.app j)
      map := fun {j j'} φ =>
        equalizer.lift (Limits.equalizer.ι (𝟙 _) (p.app j) ≫ F.map φ)
          (by rw [comp_id, assoc, p.naturality φ, ← assoc, ← Limits.equalizer.condition, comp_id]) }
  let i : Y ⟶ F :=
    { app := fun j => equalizer.ι _ _
      naturality := fun _ _ _ => by rw [equalizer.lift_ι] }
  let e : F ⟶ Y :=
    { app := fun j =>
        equalizer.lift (p.app j) (by simpa only [comp_id] using (congr_app hp j).symm)
      naturality := fun j j' φ => equalizer.hom_ext (by simp [Y]) }
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} J
    inst✝¹ : CategoryTheory.Category.{u_3, u_2} C
    P Q : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
    f : Quiver.Hom P Q
    X : J
    inst✝ : CategoryTheory.IsIdempotentComplete C
    F : CategoryTheory.Functor J C
    p : Quiver.Hom F F
    hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
    hC : ∀ (X : C) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp p  …
    this : ∀ (j : J), CategoryTheory.Limits.HasEqualizer (CategoryTheory.CategoryS …
    Y : CategoryTheory.Functor J C := { obj := fun j => CategoryTheory.Limits.equa …
    i : Quiver.Hom Y F := { app := fun j => CategoryTheory.Limits.equalizer.ι (Cat …
    e : Quiver.Hom F Y := { app := fun j => CategoryTheory.Limits.equalizer.lift ( …
    ⊢ Exists fun Y => Exists fun i => Exists fun e => And (Eq (CategoryTheory.Cate …
  -/
  use Y, i, e
  /-
    case h
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} J
    inst✝¹ : CategoryTheory.Category.{u_3, u_2} C
    P Q : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
    f : Quiver.Hom P Q
    X : J
    inst✝ : CategoryTheory.IsIdempotentComplete C
    F : CategoryTheory.Functor J C
    p : Quiver.Hom F F
    hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
    hC : ∀ (X : C) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp p  …
    this : ∀ (j : J), CategoryTheory.Limits.HasEqualizer (CategoryTheory.CategoryS …
    Y : CategoryTheory.Functor J C := { obj := fun j => CategoryTheory.Limits.equa …
    i : Quiver.Hom Y F := { app := fun j => CategoryTheory.Limits.equalizer.ι (Cat …
    e : Quiver.Hom F Y := { app := fun j => CategoryTheory.Limits.equalizer.lift ( …
    ⊢ And (Eq (CategoryTheory.CategoryStruct.comp i e) (CategoryTheory.CategoryStr …
  -/
  constructor
    /-
      case h.left
      J : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} J
      inst✝¹ : CategoryTheory.Category.{u_3, u_2} C
      P Q : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
      f : Quiver.Hom P Q
      X : J
      inst✝ : CategoryTheory.IsIdempotentComplete C
      F : CategoryTheory.Functor J C
      p : Quiver.Hom F F
      hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
      hC : ∀ (X : C) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp p  …
      this : ∀ (j : J), CategoryTheory.Limits.HasEqualizer (CategoryTheory.CategoryS …
      Y : CategoryTheory.Functor J C := { obj := fun j => CategoryTheory.Limits.equa …
      i : Quiver.Hom Y F := { app := fun j => CategoryTheory.Limits.equalizer.ι (Cat …
      e : Quiver.Hom F Y := { app := fun j => CategoryTheory.Limits.equalizer.lift ( …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp i e) (CategoryTheory.CategoryStruct.i …
    -/
  · ext j
    /-
      case h.left.w.h.h
      J : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} J
      inst✝¹ : CategoryTheory.Category.{u_3, u_2} C
      P Q : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
      f : Quiver.Hom P Q
      X : J
      inst✝ : CategoryTheory.IsIdempotentComplete C
      F : CategoryTheory.Functor J C
      p : Quiver.Hom F F
      hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
      hC : ∀ (X : C) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp p  …
      this : ∀ (j : J), CategoryTheory.Limits.HasEqualizer (CategoryTheory.CategoryS …
      Y : CategoryTheory.Functor J C := { obj := fun j => CategoryTheory.Limits.equa …
      i : Quiver.Hom Y F := { app := fun j => CategoryTheory.Limits.equalizer.ι (Cat …
      e : Quiver.Hom F Y := { app := fun j => CategoryTheory.Limits.equalizer.lift ( …
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.CategoryStruct.comp  …
    -/
    dsimp
    /-
      case h.left.w.h.h
      J : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} J
      inst✝¹ : CategoryTheory.Category.{u_3, u_2} C
      P Q : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
      f : Quiver.Hom P Q
      X : J
      inst✝ : CategoryTheory.IsIdempotentComplete C
      F : CategoryTheory.Functor J C
      p : Quiver.Hom F F
      hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
      hC : ∀ (X : C) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp p  …
      this : ∀ (j : J), CategoryTheory.Limits.HasEqualizer (CategoryTheory.CategoryS …
      Y : CategoryTheory.Functor J C := { obj := fun j => CategoryTheory.Limits.equa …
      i : Quiver.Hom Y F := { app := fun j => CategoryTheory.Limits.equalizer.ι (Cat …
      e : Quiver.Hom F Y := { app := fun j => CategoryTheory.Limits.equalizer.lift ( …
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [assoc, equalizer.lift_ι, ← equalizer.condition, id_comp, comp_id]
    /-
      🎉 no goals
    -/
    /-
      case h.right
      J : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} J
      inst✝¹ : CategoryTheory.Category.{u_3, u_2} C
      P Q : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
      f : Quiver.Hom P Q
      X : J
      inst✝ : CategoryTheory.IsIdempotentComplete C
      F : CategoryTheory.Functor J C
      p : Quiver.Hom F F
      hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
      hC : ∀ (X : C) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp p  …
      this : ∀ (j : J), CategoryTheory.Limits.HasEqualizer (CategoryTheory.CategoryS …
      Y : CategoryTheory.Functor J C := { obj := fun j => CategoryTheory.Limits.equa …
      i : Quiver.Hom Y F := { app := fun j => CategoryTheory.Limits.equalizer.ι (Cat …
      e : Quiver.Hom F Y := { app := fun j => CategoryTheory.Limits.equalizer.lift ( …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp e i) p
    -/
  · ext j
    /-
      case h.right.w.h
      J : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} J
      inst✝¹ : CategoryTheory.Category.{u_3, u_2} C
      P Q : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
      f : Quiver.Hom P Q
      X : J
      inst✝ : CategoryTheory.IsIdempotentComplete C
      F : CategoryTheory.Functor J C
      p : Quiver.Hom F F
      hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
      hC : ∀ (X : C) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp p  …
      this : ∀ (j : J), CategoryTheory.Limits.HasEqualizer (CategoryTheory.CategoryS …
      Y : CategoryTheory.Functor J C := { obj := fun j => CategoryTheory.Limits.equa …
      i : Quiver.Hom Y F := { app := fun j => CategoryTheory.Limits.equalizer.ι (Cat …
      e : Quiver.Hom F Y := { app := fun j => CategoryTheory.Limits.equalizer.lift ( …
      j : J
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp e i).app j) (p.app j)
    -/
    simp [Y, i, e]
    /-
      🎉 no goals
    -/

/-- On objects, the functor which sends a formal direct factor `P` of a
functor `F : J ⥤ C` to the functor `J ⥤ Karoubi C` which sends `(j : J)` to
the corresponding direct factor of `F.obj j`. -/
@[simps]
def obj (P : Karoubi (J ⥤ C)) : J ⥤ Karoubi C where
  obj j := ⟨P.X.obj j, P.p.app j, congr_app P.idem j⟩
  map {j j'} φ :=
    { f := P.p.app j ≫ P.X.map φ
      comm := by
        /-
          J : Type u_1
          C : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.51042, u_1} J
          inst✝ : CategoryTheory.Category.{?u.51046, u_2} C
          P✝ Q : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
          f : Quiver.Hom P✝ Q
          X : J
          P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
          j j' : J
          φ : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.p.app j) (P.X.map φ)) (CategoryThe …
        -/
        simp only [NatTrans.naturality, assoc]
        /-
          J : Type u_1
          C : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.51042, u_1} J
          inst✝ : CategoryTheory.Category.{?u.51046, u_2} C
          P✝ Q : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
          f : Quiver.Hom P✝ Q
          X : J
          P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
          j j' : J
          φ : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.p.app j) (P.X.map φ)) (CategoryThe …
        -/
        have h := congr_app P.idem j
        /-
          J : Type u_1
          C : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.51042, u_1} J
          inst✝ : CategoryTheory.Category.{?u.51046, u_2} C
          P✝ Q : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
          f : Quiver.Hom P✝ Q
          X : J
          P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
          j j' : J
          φ : Quiver.Hom j j'
          h : Eq ((CategoryTheory.CategoryStruct.comp P.p P.p).app j) (P.p.app j)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.p.app j) (P.X.map φ)) (CategoryThe …
        -/
        rw [NatTrans.comp_app] at h
        /-
          J : Type u_1
          C : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.51042, u_1} J
          inst✝ : CategoryTheory.Category.{?u.51046, u_2} C
          P✝ Q : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
          f : Quiver.Hom P✝ Q
          X : J
          P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
          j j' : J
          φ : Quiver.Hom j j'
          h : Eq (CategoryTheory.CategoryStruct.comp (P.p.app j) (P.p.app j)) (P.p.app j)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.p.app j) (P.X.map φ)) (CategoryThe …
        -/
        rw [reassoc_of% h, reassoc_of% h] }
        /-
          🎉 no goals
        -/


/-- Tautological action on maps of the functor `Karoubi (J ⥤ C) ⥤ (J ⥤ Karoubi C)`. -/
@[simps]
def map {P Q : Karoubi (J ⥤ C)} (f : P ⟶ Q) : obj P ⟶ obj Q where
  app j := ⟨f.f.app j, congr_app f.comm j⟩


/-- The tautological fully faithful functor `Karoubi (J ⥤ C) ⥤ (J ⥤ Karoubi C)`. -/
@[simps]
def karoubiFunctorCategoryEmbedding : Karoubi (J ⥤ C) ⥤ J ⥤ Karoubi C where
  obj := KaroubiFunctorCategoryEmbedding.obj
  map := KaroubiFunctorCategoryEmbedding.map


instance : (karoubiFunctorCategoryEmbedding J C).Full where
  map_surjective {P Q} f :=
   ⟨{ f :=
        { app := fun j => (f.app j).f
          naturality := fun j j' φ => by
            /-
              J : Type u_1
              C : Type u_2
              inst✝¹ : CategoryTheory.Category.{u_4, u_1} J
              inst✝ : CategoryTheory.Category.{u_3, u_2} C
              P✝ Q✝ : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
              f✝ : Quiver.Hom P✝ Q✝
              X : J
              P Q : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
              f : Quiver.Hom ((CategoryTheory.Idempotents.karoubiFunctorCategoryEmbedding J  …
              j j' : J
              φ : Quiver.Hom j j'
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.X.map φ) ((fun j => (f.app j).f) j …
            -/
            rw [← Karoubi.comp_p_assoc]
            /-
              J : Type u_1
              C : Type u_2
              inst✝¹ : CategoryTheory.Category.{u_4, u_1} J
              inst✝ : CategoryTheory.Category.{u_3, u_2} C
              P✝ Q✝ : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
              f✝ : Quiver.Hom P✝ Q✝
              X : J
              P Q : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
              f : Quiver.Hom ((CategoryTheory.Idempotents.karoubiFunctorCategoryEmbedding J  …
              j j' : J
              φ : Quiver.Hom j j'
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.X.map φ) ((fun j => (f.app j).f) j …
            -/
            have h := hom_ext_iff.mp (f.naturality φ)
            /-
              J : Type u_1
              C : Type u_2
              inst✝¹ : CategoryTheory.Category.{u_4, u_1} J
              inst✝ : CategoryTheory.Category.{u_3, u_2} C
              P✝ Q✝ : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
              f✝ : Quiver.Hom P✝ Q✝
              X : J
              P Q : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
              f : Quiver.Hom ((CategoryTheory.Idempotents.karoubiFunctorCategoryEmbedding J  …
              j j' : J
              φ : Quiver.Hom j j'
              h : Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Idempotents.karou …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.X.map φ) ((fun j => (f.app j).f) j …
            -/
            simp only [comp_f] at h
            /-
              J : Type u_1
              C : Type u_2
              inst✝¹ : CategoryTheory.Category.{u_4, u_1} J
              inst✝ : CategoryTheory.Category.{u_3, u_2} C
              P✝ Q✝ : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
              f✝ : Quiver.Hom P✝ Q✝
              X : J
              P Q : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
              f : Quiver.Hom ((CategoryTheory.Idempotents.karoubiFunctorCategoryEmbedding J  …
              j j' : J
              φ : Quiver.Hom j j'
              h : Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Idempotents.karou …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.X.map φ) ((fun j => (f.app j).f) j …
            -/
            dsimp [karoubiFunctorCategoryEmbedding] at h
            /-
              J : Type u_1
              C : Type u_2
              inst✝¹ : CategoryTheory.Category.{u_4, u_1} J
              inst✝ : CategoryTheory.Category.{u_3, u_2} C
              P✝ Q✝ : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
              f✝ : Quiver.Hom P✝ Q✝
              X : J
              P Q : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
              f : Quiver.Hom ((CategoryTheory.Idempotents.karoubiFunctorCategoryEmbedding J  …
              j j' : J
              φ : Quiver.Hom j j'
              h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.X.map φ) ((fun j => (f.app j).f) j …
            -/
            erw [← h, assoc, ← P.p.naturality_assoc φ, p_comp (f.app j')] }
            /-
              🎉 no goals
            -/
      comm := by
        /-
          J : Type u_1
          C : Type u_2
          inst✝¹ : CategoryTheory.Category.{u_4, u_1} J
          inst✝ : CategoryTheory.Category.{u_3, u_2} C
          P✝ Q✝ : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
          f✝ : Quiver.Hom P✝ Q✝
          X : J
          P Q : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
          f : Quiver.Hom ((CategoryTheory.Idempotents.karoubiFunctorCategoryEmbedding J  …
          ⊢ Eq { app := fun j => (f.app j).f, naturality := ⋯ } (CategoryTheory.Category …
        -/
        ext j
        /-
          case w.h
          J : Type u_1
          C : Type u_2
          inst✝¹ : CategoryTheory.Category.{u_4, u_1} J
          inst✝ : CategoryTheory.Category.{u_3, u_2} C
          P✝ Q✝ : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
          f✝ : Quiver.Hom P✝ Q✝
          X : J
          P Q : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
          f : Quiver.Hom ((CategoryTheory.Idempotents.karoubiFunctorCategoryEmbedding J  …
          j : J
          ⊢ Eq ({ app := fun j => (f.app j).f, naturality := ⋯ }.app j) ((CategoryTheory …
        -/
        exact (f.app j).comm }, rfl⟩
        /-
          🎉 no goals
        -/


instance : (karoubiFunctorCategoryEmbedding J C).Faithful where
  map_injective h := by
    /-
      J : Type u_1
      C : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} J
      inst✝ : CategoryTheory.Category.{u_3, u_2} C
      P Q : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
      f : Quiver.Hom P Q
      X : J
      X✝ Y✝ : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
      a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
      h : Eq ((CategoryTheory.Idempotents.karoubiFunctorCategoryEmbedding J C).map a …
      ⊢ Eq a₁✝ a₂✝
    -/
    ext j
    /-
      case h.w.h
      J : Type u_1
      C : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} J
      inst✝ : CategoryTheory.Category.{u_3, u_2} C
      P Q : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
      f : Quiver.Hom P Q
      X : J
      X✝ Y✝ : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Functor J C)
      a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
      h : Eq ((CategoryTheory.Idempotents.karoubiFunctorCategoryEmbedding J C).map a …
      j : J
      ⊢ Eq (a₁✝.f.app j) (a₂✝.f.app j)
    -/
    exact hom_ext_iff.mp (congr_app h j)
    /-
      🎉 no goals
    -/


/-- The composition of `(J ⥤ C) ⥤ Karoubi (J ⥤ C)` and `Karoubi (J ⥤ C) ⥤ (J ⥤ Karoubi C)`
equals the functor `(J ⥤ C) ⥤ (J ⥤ Karoubi C)` given by the composition with
`toKaroubi C : C ⥤ Karoubi C`. -/
theorem toKaroubi_comp_karoubiFunctorCategoryEmbedding :
    toKaroubi _ ⋙ karoubiFunctorCategoryEmbedding J C =
      (whiskeringRight J _ _).obj (toKaroubi C) := by
  /-
    J : Type u_1
    C : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} J
    inst✝ : CategoryTheory.Category.{u_4, u_2} C
    ⊢ Eq ((CategoryTheory.Idempotents.toKaroubi (CategoryTheory.Functor J C)).comp …
  -/
  apply Functor.ext
    /-
      case h_map
      J : Type u_1
      C : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} J
      inst✝ : CategoryTheory.Category.{u_4, u_2} C
      ⊢ autoParam (∀ (X Y : CategoryTheory.Functor J C) (f : Quiver.Hom X Y), Eq ((( …
    -/
  · intro X Y f
    /-
      case h_map
      J : Type u_1
      C : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} J
      inst✝ : CategoryTheory.Category.{u_4, u_2} C
      X Y : CategoryTheory.Functor J C
      f : Quiver.Hom X Y
      ⊢ Eq (((CategoryTheory.Idempotents.toKaroubi (CategoryTheory.Functor J C)).com …
    -/
    ext j
    /-
      case h_map.w.h.h
      J : Type u_1
      C : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} J
      inst✝ : CategoryTheory.Category.{u_4, u_2} C
      X Y : CategoryTheory.Functor J C
      f : Quiver.Hom X Y
      j : J
      ⊢ Eq ((((CategoryTheory.Idempotents.toKaroubi (CategoryTheory.Functor J C)).co …
    -/
    dsimp [toKaroubi]
    /-
      case h_map.w.h.h
      J : Type u_1
      C : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} J
      inst✝ : CategoryTheory.Category.{u_4, u_2} C
      X Y : CategoryTheory.Functor J C
      f : Quiver.Hom X Y
      j : J
      ⊢ Eq (f.app j) (CategoryTheory.CategoryStruct.comp ((CategoryTheory.eqToHom ⋯) …
    -/
    simp only [eqToHom_app, eqToHom_refl]
    /-
      case h_map.w.h.h
      J : Type u_1
      C : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} J
      inst✝ : CategoryTheory.Category.{u_4, u_2} C
      X Y : CategoryTheory.Functor J C
      f : Quiver.Hom X Y
      j : J
      ⊢ Eq (f.app j) (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStr …
    -/
    erw [comp_id, id_comp]
    /-
      🎉 no goals
    -/
    /-
      case h_obj
      J : Type u_1
      C : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} J
      inst✝ : CategoryTheory.Category.{u_4, u_2} C
      ⊢ ∀ (X : CategoryTheory.Functor J C), Eq (((CategoryTheory.Idempotents.toKarou …
    -/
  · intro X
    /-
      case h_obj
      J : Type u_1
      C : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} J
      inst✝ : CategoryTheory.Category.{u_4, u_2} C
      X : CategoryTheory.Functor J C
      ⊢ Eq (((CategoryTheory.Idempotents.toKaroubi (CategoryTheory.Functor J C)).com …
    -/
    apply Functor.ext
      /-
        case h_obj.h_map
        J : Type u_1
        C : Type u_2
        inst✝¹ : CategoryTheory.Category.{u_3, u_1} J
        inst✝ : CategoryTheory.Category.{u_4, u_2} C
        X : CategoryTheory.Functor J C
        ⊢ autoParam (∀ (X_1 Y : J) (f : Quiver.Hom X_1 Y), Eq ((((CategoryTheory.Idemp …
      -/
    · intro j j' φ
      /-
        case h_obj.h_map
        J : Type u_1
        C : Type u_2
        inst✝¹ : CategoryTheory.Category.{u_3, u_1} J
        inst✝ : CategoryTheory.Category.{u_4, u_2} C
        X : CategoryTheory.Functor J C
        j j' : J
        φ : Quiver.Hom j j'
        ⊢ Eq ((((CategoryTheory.Idempotents.toKaroubi (CategoryTheory.Functor J C)).co …
      -/
      ext
      /-
        case h_obj.h_map.h
        J : Type u_1
        C : Type u_2
        inst✝¹ : CategoryTheory.Category.{u_3, u_1} J
        inst✝ : CategoryTheory.Category.{u_4, u_2} C
        X : CategoryTheory.Functor J C
        j j' : J
        φ : Quiver.Hom j j'
        ⊢ Eq ((((CategoryTheory.Idempotents.toKaroubi (CategoryTheory.Functor J C)).co …
      -/
      dsimp
      /-
        case h_obj.h_map.h
        J : Type u_1
        C : Type u_2
        inst✝¹ : CategoryTheory.Category.{u_3, u_1} J
        inst✝ : CategoryTheory.Category.{u_4, u_2} C
        X : CategoryTheory.Functor J C
        j j' : J
        φ : Quiver.Hom j j'
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (X. …
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case h_obj.h_obj
        J : Type u_1
        C : Type u_2
        inst✝¹ : CategoryTheory.Category.{u_3, u_1} J
        inst✝ : CategoryTheory.Category.{u_4, u_2} C
        X : CategoryTheory.Functor J C
        ⊢ ∀ (X_1 : J), Eq ((((CategoryTheory.Idempotents.toKaroubi (CategoryTheory.Fun …
      -/
    · intro j
      /-
        case h_obj.h_obj
        J : Type u_1
        C : Type u_2
        inst✝¹ : CategoryTheory.Category.{u_3, u_1} J
        inst✝ : CategoryTheory.Category.{u_4, u_2} C
        X : CategoryTheory.Functor J C
        j : J
        ⊢ Eq ((((CategoryTheory.Idempotents.toKaroubi (CategoryTheory.Functor J C)).co …
      -/
      rfl
      /-
        🎉 no goals
      -/


