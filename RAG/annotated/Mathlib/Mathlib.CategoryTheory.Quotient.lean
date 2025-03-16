/-- A `HomRel` on `C` consists of a relation on every hom-set. -/
def HomRel (C) [Quiver C] :=
  ∀ ⦃X Y : C⦄, (X ⟶ Y) → (X ⟶ Y) → Prop

-- Porting Note: `deriving Inhabited` was not able to deduce this typeclass

instance (C) [Quiver C] : Inhabited (HomRel C) where
  default := fun _ _ _ _ ↦ PUnit


/-- A functor induces a `HomRel` on its domain, relating those maps that have the same image. -/
def Functor.homRel : HomRel C :=
  fun _ _ f g ↦ F.map f = F.map g


@[simp]
lemma Functor.homRel_iff {X Y : C} (f g : X ⟶ Y) :
    F.homRel f g ↔ F.map f = F.map g := Iff.rfl


/-- A `HomRel` is a congruence when it's an equivalence on every hom-set, and it can be composed
from left and right. -/
class Congruence : Prop where
  /-- `r` is an equivalence on every hom-set. -/
  equivalence : ∀ {X Y}, _root_.Equivalence (@r X Y)
  /-- Precomposition with an arrow respects `r`. -/
  compLeft : ∀ {X Y Z} (f : X ⟶ Y) {g g' : Y ⟶ Z}, r g g' → r (f ≫ g) (f ≫ g')
  /-- Postcomposition with an arrow respects `r`. -/
  compRight : ∀ {X Y Z} {f f' : X ⟶ Y} (g : Y ⟶ Z), r f f' → r (f ≫ g) (f' ≫ g)


/-- For `F : C ⥤ D`, `F.homRel` is a congruence.-/
instance Functor.congruence_homRel {C D : Type*} [Category C] [Category D] (F : C ⥤ D) :
    Congruence F.homRel where
  equivalence :=
    { refl := fun _ ↦ rfl
                 /-
                   C✝ : Type ?u.1140
                   inst✝² : CategoryTheory.Category.{?u.1144, ?u.1140} C✝
                   r : HomRel C✝
                   C : Type u_1
                   D : Type u_2
                   inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
                   inst✝ : CategoryTheory.Category.{u_4, u_2} D
                   F : CategoryTheory.Functor C D
                   X✝ Y✝ : C
                   ⊢ ∀ {x y : Quiver.Hom X✝ Y✝}, F.homRel x y → F.homRel y x
                 -/
      symm := by aesop
                 /-
                   🎉 no goals
                 -/
                  /-
                    C✝ : Type ?u.1140
                    inst✝² : CategoryTheory.Category.{?u.1144, ?u.1140} C✝
                    r : HomRel C✝
                    C : Type u_1
                    D : Type u_2
                    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
                    inst✝ : CategoryTheory.Category.{u_4, u_2} D
                    F : CategoryTheory.Functor C D
                    X✝ Y✝ : C
                    ⊢ ∀ {x y z : Quiver.Hom X✝ Y✝}, F.homRel x y → F.homRel y z → F.homRel x z
                  -/
      trans := by aesop }
                  /-
                    🎉 no goals
                  -/
                 /-
                   C✝ : Type ?u.1140
                   inst✝² : CategoryTheory.Category.{?u.1144, ?u.1140} C✝
                   r : HomRel C✝
                   C : Type u_1
                   D : Type u_2
                   inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
                   inst✝ : CategoryTheory.Category.{u_4, u_2} D
                   F : CategoryTheory.Functor C D
                   ⊢ ∀ {X Y Z : C} (f : Quiver.Hom X Y) {g g' : Quiver.Hom Y Z}, F.homRel g g' →  …
                 -/
  compLeft := by aesop
                 /-
                   🎉 no goals
                 -/
                  /-
                    C✝ : Type ?u.1140
                    inst✝² : CategoryTheory.Category.{?u.1144, ?u.1140} C✝
                    r : HomRel C✝
                    C : Type u_1
                    D : Type u_2
                    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
                    inst✝ : CategoryTheory.Category.{u_4, u_2} D
                    F : CategoryTheory.Functor C D
                    ⊢ ∀ {X Y Z : C} {f f' : Quiver.Hom X Y} (g : Quiver.Hom Y Z), F.homRel f f' →  …
                  -/
  compRight := by aesop
                  /-
                    🎉 no goals
                  -/


/-- A type synonym for `C`, thought of as the objects of the quotient category. -/
@[ext]
structure Quotient (r : HomRel C) where
  /-- The object of `C`. -/
  as : C


instance [Inhabited C] : Inhabited (Quotient r) :=
  ⟨{ as := default }⟩


/-- Generates the closure of a family of relations w.r.t. composition from left and right. -/
inductive CompClosure (r : HomRel C) ⦃s t : C⦄ : (s ⟶ t) → (s ⟶ t) → Prop
  | intro {a b : C} (f : s ⟶ a) (m₁ m₂ : a ⟶ b) (g : b ⟶ t) (h : r m₁ m₂) :
    CompClosure r (f ≫ m₁ ≫ g) (f ≫ m₂ ≫ g)


theorem CompClosure.of {a b : C} (m₁ m₂ : a ⟶ b) (h : r m₁ m₂) : CompClosure r m₁ m₂ := by
  /-
    C : Type u_2
    inst✝ : CategoryTheory.Category.{u_1, u_2} C
    r : HomRel C
    a b : C
    m₁ m₂ : Quiver.Hom a b
    h : r m₁ m₂
    ⊢ CategoryTheory.Quotient.CompClosure r m₁ m₂
  -/
  simpa using CompClosure.intro (𝟙 _) m₁ m₂ (𝟙 _) h
  /-
    🎉 no goals
  -/


theorem comp_left {a b c : C} (f : a ⟶ b) :
    ∀ (g₁ g₂ : b ⟶ c) (_ : CompClosure r g₁ g₂), CompClosure r (f ≫ g₁) (f ≫ g₂)
                                  /-
                                    C : Type u_2
                                    inst✝ : CategoryTheory.Category.{u_1, u_2} C
                                    r : HomRel C
                                    a b c : C
                                    f : Quiver.Hom a b
                                    a✝ b✝ : C
                                    x : Quiver.Hom b a✝
                                    m₁ m₂ : Quiver.Hom a✝ b✝
                                    y : Quiver.Hom b✝ c
                                    h : r m₁ m₂
                                    ⊢ CategoryTheory.Quotient.CompClosure r (CategoryTheory.CategoryStruct.comp f  …
                                  -/
  | _, _, ⟨x, m₁, m₂, y, h⟩ => by simpa using CompClosure.intro (f ≫ x) m₁ m₂ y h
                                  /-
                                    🎉 no goals
                                  -/


theorem comp_right {a b c : C} (g : b ⟶ c) :
    ∀ (f₁ f₂ : a ⟶ b) (_ : CompClosure r f₁ f₂), CompClosure r (f₁ ≫ g) (f₂ ≫ g)
                                  /-
                                    C : Type u_2
                                    inst✝ : CategoryTheory.Category.{u_1, u_2} C
                                    r : HomRel C
                                    a b c : C
                                    g : Quiver.Hom b c
                                    a✝ b✝ : C
                                    x : Quiver.Hom a a✝
                                    m₁ m₂ : Quiver.Hom a✝ b✝
                                    y : Quiver.Hom b✝ b
                                    h : r m₁ m₂
                                    ⊢ CategoryTheory.Quotient.CompClosure r (CategoryTheory.CategoryStruct.comp (C …
                                  -/
  | _, _, ⟨x, m₁, m₂, y, h⟩ => by simpa using CompClosure.intro x m₁ m₂ (y ≫ g) h
                                  /-
                                    🎉 no goals
                                  -/


/-- Hom-sets of the quotient category. -/
def Hom (s t : Quotient r) :=
  Quot <| @CompClosure C _ r s.as t.as


instance (a : Quotient r) : Inhabited (Hom r a a) :=
  ⟨Quot.mk _ (𝟙 a.as)⟩


/-- Composition in the quotient category. -/
def comp ⦃a b c : Quotient r⦄ : Hom r a b → Hom r b c → Hom r a c := fun hf hg ↦
  Quot.liftOn hf
    (fun f ↦
      Quot.liftOn hg (fun g ↦ Quot.mk _ (f ≫ g)) fun g₁ g₂ h ↦
        Quot.sound <| comp_left r f g₁ g₂ h)
    fun f₁ f₂ h ↦ Quot.inductionOn hg fun g ↦ Quot.sound <| comp_right r g f₁ f₂ h


@[simp]
theorem comp_mk {a b c : Quotient r} (f : a.as ⟶ b.as) (g : b.as ⟶ c.as) :
    comp r (Quot.mk _ f) (Quot.mk _ g) = Quot.mk _ (f ≫ g) :=
  rfl

-- Porting note: Had to manually add the proofs of `comp_id` `id_comp` and `assoc`

instance category : Category (Quotient r) where
  Hom := Hom r
  id a := Quot.mk _ (𝟙 a.as)
  comp := @comp _ _ r
                                        /-
                                          C : Type ?u.8003
                                          inst✝ : CategoryTheory.Category.{?u.8007, ?u.8003} C
                                          r : HomRel C
                                          X✝ Y✝ : CategoryTheory.Quotient r
                                          f : Quiver.Hom X✝ Y✝
                                          ⊢ ∀ (a : Quiver.Hom X✝.as Y✝.as), Eq (CategoryTheory.CategoryStruct.comp (Quot …
                                        -/
                                        /-
                                          C : Type ?u.8003
                                          inst✝ : CategoryTheory.Category.{?u.8007, ?u.8003} C
                                          r : HomRel C
                                          X✝ Y✝ : CategoryTheory.Quotient r
                                          f : Quiver.Hom X✝ Y✝
                                          ⊢ ∀ (a : Quiver.Hom X✝.as Y✝.as), Eq (CategoryTheory.CategoryStruct.comp (Cate …
                                        -/
  comp_id f := Quot.inductionOn f <| by simp
                                        /-
                                          🎉 no goals
                                        -/
                                        /-
                                          🎉 no goals
                                        -/
  id_comp f := Quot.inductionOn f <| by simp
                                                                                      /-
                                                                                        C : Type ?u.8003
                                                                                        inst✝ : CategoryTheory.Category.{?u.8007, ?u.8003} C
                                                                                        r : HomRel C
                                                                                        W✝ X✝ Y✝ Z✝ : CategoryTheory.Quotient r
                                                                                        f : Quiver.Hom W✝ X✝
                                                                                        g : Quiver.Hom X✝ Y✝
                                                                                        h : Quiver.Hom Y✝ Z✝
                                                                                        ⊢ ∀ (a : Quiver.Hom Y✝.as Z✝.as) (a_1 : Quiver.Hom X✝.as Y✝.as) (a_2 : Quiver. …
                                                                                      -/
  assoc f g h := Quot.inductionOn f <| Quot.inductionOn g <| Quot.inductionOn h <| by simp
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


/-- The functor from a category to its quotient. -/
def functor : C ⥤ Quotient r where
  obj a := { as := a }
  map := @fun _ _ f ↦ Quot.mk _ f


instance full_functor : (functor r).Full where
                                      /-
                                        C : Type u_2
                                        inst✝ : CategoryTheory.Category.{u_1, u_2} C
                                        r : HomRel C
                                        X✝ Y✝ : C
                                        f : Quiver.Hom ((CategoryTheory.Quotient.functor r).obj X✝) ((CategoryTheory.Q …
                                        ⊢ Eq ((CategoryTheory.Quotient.functor r).map (Quot.out f)) f
                                      -/
  map_surjective f := ⟨Quot.out f, by simp [functor]⟩
                                      /-
                                        🎉 no goals
                                      -/


instance essSurj_functor : (functor r).EssSurj where
  mem_essImage Y :=
    ⟨Y.as, ⟨eqToIso (by
            /-
              C : Type u_2
              inst✝ : CategoryTheory.Category.{u_1, u_2} C
              r : HomRel C
              Y : CategoryTheory.Quotient r
              ⊢ Eq ((CategoryTheory.Quotient.functor r).obj Y.as) Y
            -/
            ext
            /-
              case as
              C : Type u_2
              inst✝ : CategoryTheory.Category.{u_1, u_2} C
              r : HomRel C
              Y : CategoryTheory.Quotient r
              ⊢ Eq ((CategoryTheory.Quotient.functor r).obj Y.as).as Y.as
            -/
            rfl)⟩⟩
            /-
              🎉 no goals
            -/


protected theorem induction {P : ∀ {a b : Quotient r}, (a ⟶ b) → Prop}
    (h : ∀ {x y : C} (f : x ⟶ y), P ((functor r).map f)) :
    ∀ {a b : Quotient r} (f : a ⟶ b), P f := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    r : HomRel C
    P : {a b : CategoryTheory.Quotient r} → Quiver.Hom a b → Prop
    h : ∀ {x y : C} (f : Quiver.Hom x y), P ((CategoryTheory.Quotient.functor r).m …
    ⊢ ∀ {a b : CategoryTheory.Quotient r} (f : Quiver.Hom a b), P f
  -/
  rintro ⟨x⟩ ⟨y⟩ ⟨f⟩
  /-
    case mk.mk.mk
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    r : HomRel C
    P : {a b : CategoryTheory.Quotient r} → Quiver.Hom a b → Prop
    h : ∀ {x y : C} (f : Quiver.Hom x y), P ((CategoryTheory.Quotient.functor r).m …
    x y : C
    f✝ : Quiver.Hom { as := x } { as := y }
    f : Quiver.Hom { as := x }.as { as := y }.as
    ⊢ P (Quot.mk (CategoryTheory.Quotient.CompClosure r) f)
  -/
  exact h f
  /-
    🎉 no goals
  -/


protected theorem sound {a b : C} {f₁ f₂ : a ⟶ b} (h : r f₁ f₂) :
    (functor r).map f₁ = (functor r).map f₂ := by
  /-
    C : Type u_2
    inst✝ : CategoryTheory.Category.{u_1, u_2} C
    r : HomRel C
    a b : C
    f₁ f₂ : Quiver.Hom a b
    h : r f₁ f₂
    ⊢ Eq ((CategoryTheory.Quotient.functor r).map f₁) ((CategoryTheory.Quotient.fu …
  -/
  simpa using Quot.sound (CompClosure.intro (𝟙 a) f₁ f₂ (𝟙 b) h)
  /-
    🎉 no goals
  -/


lemma compClosure_iff_self [h : Congruence r] {X Y : C} (f g : X ⟶ Y) :
    CompClosure r f g ↔ r f g := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    r : HomRel C
    h : CategoryTheory.Congruence r
    X Y : C
    f g : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Quotient.CompClosure r f g) (r f g)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      r : HomRel C
      h : CategoryTheory.Congruence r
      X Y : C
      f g : Quiver.Hom X Y
      ⊢ CategoryTheory.Quotient.CompClosure r f g → r f g
    -/
  · intro hfg
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      r : HomRel C
      h : CategoryTheory.Congruence r
      X Y : C
      f g : Quiver.Hom X Y
      hfg : CategoryTheory.Quotient.CompClosure r f g
      ⊢ r f g
    -/
    induction' hfg with m m' hm
    /-
      case mp.intro
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      r : HomRel C
      h : CategoryTheory.Congruence r
      X Y : C
      f g : Quiver.Hom X Y
      m m' : C
      hm : Quiver.Hom X m
      m₁✝ m₂✝ : Quiver.Hom m m'
      g✝ : Quiver.Hom m' Y
      h✝ : r m₁✝ m₂✝
      ⊢ r (CategoryTheory.CategoryStruct.comp hm (CategoryTheory.CategoryStruct.comp …
    -/
    exact Congruence.compLeft _ (Congruence.compRight _ (by assumption))
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      r : HomRel C
      h : CategoryTheory.Congruence r
      X Y : C
      f g : Quiver.Hom X Y
      ⊢ r f g → CategoryTheory.Quotient.CompClosure r f g
    -/
  · exact CompClosure.of _ _ _
    /-
      🎉 no goals
    -/


@[simp]
theorem compClosure_eq_self [h : Congruence r] :
    CompClosure r = r := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    r : HomRel C
    h : CategoryTheory.Congruence r
    ⊢ Eq (CategoryTheory.Quotient.CompClosure r) r
  -/
  ext
  /-
    case h.h.h.h.a
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    r : HomRel C
    h : CategoryTheory.Congruence r
    x✝³ x✝² : C
    x✝¹ x✝ : Quiver.Hom x✝³ x✝²
    ⊢ Iff (CategoryTheory.Quotient.CompClosure r x✝¹ x✝) (r x✝¹ x✝)
  -/
  simp only [compClosure_iff_self]
  /-
    🎉 no goals
  -/


theorem functor_map_eq_iff [h : Congruence r] {X Y : C} (f f' : X ⟶ Y) :
    (functor r).map f = (functor r).map f' ↔ r f f' := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    r : HomRel C
    h : CategoryTheory.Congruence r
    X Y : C
    f f' : Quiver.Hom X Y
    ⊢ Iff (Eq ((CategoryTheory.Quotient.functor r).map f) ((CategoryTheory.Quotien …
  -/
  dsimp [functor]
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    r : HomRel C
    h : CategoryTheory.Congruence r
    X Y : C
    f f' : Quiver.Hom X Y
    ⊢ Iff (Eq (Quot.mk (CategoryTheory.Quotient.CompClosure r) f) (Quot.mk (Catego …
  -/
  rw [Equivalence.quot_mk_eq_iff, compClosure_eq_self r]
  /-
    case h
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    r : HomRel C
    h : CategoryTheory.Congruence r
    X Y : C
    f f' : Quiver.Hom X Y
    ⊢ _root_.Equivalence (CategoryTheory.Quotient.CompClosure r)
  -/
  simpa only [compClosure_eq_self r] using h.equivalence
  /-
    🎉 no goals
  -/


theorem functor_homRel_eq_compClosure_eqvGen {X Y : C} (f g : X ⟶ Y) :
    (functor r).homRel f g ↔ Relation.EqvGen (@CompClosure C _ r X Y) f g :=
  Quot.eq


theorem compClosure.congruence :
    Congruence fun X Y => Relation.EqvGen (@CompClosure C _ r X Y) := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    r : HomRel C
    ⊢ CategoryTheory.Congruence fun X Y => Relation.EqvGen (CategoryTheory.Quotien …
  -/
  convert inferInstanceAs (Congruence (functor r).homRel)
  /-
    case h.e'_3.h.h
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    r : HomRel C
    x✝¹ x✝ : C
    ⊢ Eq (Relation.EqvGen (CategoryTheory.Quotient.CompClosure r)) (CategoryTheory …
  -/
  ext
  /-
    case h.e'_3.h.h.h.h.a
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    r : HomRel C
    x✝³ x✝² : C
    x✝¹ x✝ : Quiver.Hom x✝³ x✝²
    ⊢ Iff (Relation.EqvGen (CategoryTheory.Quotient.CompClosure r) x✝¹ x✝) ((Categ …
  -/
  rw [functor_homRel_eq_compClosure_eqvGen]
  /-
    🎉 no goals
  -/


/-- The induced functor on the quotient category. -/
def lift (H : ∀ (x y : C) (f₁ f₂ : x ⟶ y), r f₁ f₂ → F.map f₁ = F.map f₂) : Quotient r ⥤ D where
  obj a := F.obj a.as
  map := @fun a b hf ↦
    Quot.liftOn hf (fun f ↦ F.map f)
      (by
        /-
          C : Type ?u.16431
          inst✝¹ : CategoryTheory.Category.{?u.16435, ?u.16431} C
          r : HomRel C
          D : Type ?u.16463
          inst✝ : CategoryTheory.Category.{?u.16467, ?u.16463} D
          F : CategoryTheory.Functor C D
          H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
          a b : CategoryTheory.Quotient r
          hf : Quiver.Hom a b
          ⊢ ∀ (a_1 b_1 : Quiver.Hom a.as b.as), CategoryTheory.Quotient.CompClosure r a_ …
        -/
        rintro _ _ ⟨_, _, _, _, h⟩
        /-
          case intro
          C : Type ?u.16431
          inst✝¹ : CategoryTheory.Category.{?u.16435, ?u.16431} C
          r : HomRel C
          D : Type ?u.16463
          inst✝ : CategoryTheory.Category.{?u.16467, ?u.16463} D
          F : CategoryTheory.Functor C D
          H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
          a b : CategoryTheory.Quotient r
          hf : Quiver.Hom a b
          a✝ b✝ : C
          f✝ : Quiver.Hom a.as a✝
          m₁✝ m₂✝ : Quiver.Hom a✝ b✝
          g✝ : Quiver.Hom b✝ b.as
          h : r m₁✝ m₂✝
          ⊢ Eq ((fun f => F.map f) (CategoryTheory.CategoryStruct.comp f✝ (CategoryTheor …
        -/
        simp [H _ _ _ _ h])
        /-
          🎉 no goals
        -/
  map_id a := F.map_id a.as
  map_comp := by
    /-
      C : Type ?u.16431
      inst✝¹ : CategoryTheory.Category.{?u.16435, ?u.16431} C
      r : HomRel C
      D : Type ?u.16463
      inst✝ : CategoryTheory.Category.{?u.16467, ?u.16463} D
      F : CategoryTheory.Functor C D
      H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
      ⊢ ∀ {X Y Z : CategoryTheory.Quotient r} (f : Quiver.Hom X Y) (g : Quiver.Hom Y …
    -/
    rintro a b c ⟨f⟩ ⟨g⟩
    /-
      case mk.mk
      C : Type ?u.16431
      inst✝¹ : CategoryTheory.Category.{?u.16435, ?u.16431} C
      r : HomRel C
      D : Type ?u.16463
      inst✝ : CategoryTheory.Category.{?u.16467, ?u.16463} D
      F : CategoryTheory.Functor C D
      H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
      a b c : CategoryTheory.Quotient r
      f✝ : Quiver.Hom a b
      f : Quiver.Hom a.as b.as
      g✝ : Quiver.Hom b c
      g : Quiver.Hom b.as c.as
      ⊢ Eq ({ obj := fun a => F.obj a.as, map := fun a b hf => Quot.liftOn hf (fun f …
    -/
    exact F.map_comp f g
    /-
      🎉 no goals
    -/


theorem lift_spec : functor r ⋙ lift r F H = F := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    r : HomRel C
    D : Type u_3
    inst✝ : CategoryTheory.Category.{u_4, u_3} D
    F : CategoryTheory.Functor C D
    H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
    ⊢ Eq ((CategoryTheory.Quotient.functor r).comp (CategoryTheory.Quotient.lift r …
  -/
  apply Functor.ext; rotate_left
    /-
      case h_obj
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      r : HomRel C
      D : Type u_3
      inst✝ : CategoryTheory.Category.{u_4, u_3} D
      F : CategoryTheory.Functor C D
      H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
      ⊢ ∀ (X : C), Eq (((CategoryTheory.Quotient.functor r).comp (CategoryTheory.Quo …
    -/
  · rintro X
    /-
      case h_obj
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      r : HomRel C
      D : Type u_3
      inst✝ : CategoryTheory.Category.{u_4, u_3} D
      F : CategoryTheory.Functor C D
      H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
      X : C
      ⊢ Eq (((CategoryTheory.Quotient.functor r).comp (CategoryTheory.Quotient.lift  …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case h_map
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      r : HomRel C
      D : Type u_3
      inst✝ : CategoryTheory.Category.{u_4, u_3} D
      F : CategoryTheory.Functor C D
      H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
      ⊢ autoParam (∀ (X Y : C) (f : Quiver.Hom X Y), Eq (((CategoryTheory.Quotient.f …
    -/
  · rintro X Y f
    /-
      case h_map
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      r : HomRel C
      D : Type u_3
      inst✝ : CategoryTheory.Category.{u_4, u_3} D
      F : CategoryTheory.Functor C D
      H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
      X Y : C
      f : Quiver.Hom X Y
      ⊢ Eq (((CategoryTheory.Quotient.functor r).comp (CategoryTheory.Quotient.lift  …
    -/
    dsimp [lift, functor]
    /-
      case h_map
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      r : HomRel C
      D : Type u_3
      inst✝ : CategoryTheory.Category.{u_4, u_3} D
      F : CategoryTheory.Functor C D
      H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
      X Y : C
      f : Quiver.Hom X Y
      ⊢ Eq ((Quot.mk (CategoryTheory.Quotient.CompClosure r) f).liftOn (fun f => F.m …
    -/
    simp
    /-
      🎉 no goals
    -/


theorem lift_unique (Φ : Quotient r ⥤ D) (hΦ : functor r ⋙ Φ = F) : Φ = lift r F H := by
  /-
    C : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_1, u_3} C
    r : HomRel C
    D : Type u_4
    inst✝ : CategoryTheory.Category.{u_2, u_4} D
    F : CategoryTheory.Functor C D
    H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
    Φ : CategoryTheory.Functor (CategoryTheory.Quotient r) D
    hΦ : Eq ((CategoryTheory.Quotient.functor r).comp Φ) F
    ⊢ Eq Φ (CategoryTheory.Quotient.lift r F H)
  -/
  subst_vars
  /-
    C : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_1, u_3} C
    r : HomRel C
    D : Type u_4
    inst✝ : CategoryTheory.Category.{u_2, u_4} D
    Φ : CategoryTheory.Functor (CategoryTheory.Quotient r) D
    H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (((CategoryTheory.Quoti …
    ⊢ Eq Φ (CategoryTheory.Quotient.lift r ((CategoryTheory.Quotient.functor r).co …
  -/
  fapply Functor.hext
    /-
      case h_obj
      C : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_1, u_3} C
      r : HomRel C
      D : Type u_4
      inst✝ : CategoryTheory.Category.{u_2, u_4} D
      Φ : CategoryTheory.Functor (CategoryTheory.Quotient r) D
      H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (((CategoryTheory.Quoti …
      ⊢ ∀ (X : CategoryTheory.Quotient r), Eq (Φ.obj X) ((CategoryTheory.Quotient.li …
    -/
  · rintro X
    /-
      case h_obj
      C : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_1, u_3} C
      r : HomRel C
      D : Type u_4
      inst✝ : CategoryTheory.Category.{u_2, u_4} D
      Φ : CategoryTheory.Functor (CategoryTheory.Quotient r) D
      H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (((CategoryTheory.Quoti …
      X : CategoryTheory.Quotient r
      ⊢ Eq (Φ.obj X) ((CategoryTheory.Quotient.lift r ((CategoryTheory.Quotient.func …
    -/
    dsimp [lift, Functor]
    /-
      case h_obj
      C : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_1, u_3} C
      r : HomRel C
      D : Type u_4
      inst✝ : CategoryTheory.Category.{u_2, u_4} D
      Φ : CategoryTheory.Functor (CategoryTheory.Quotient r) D
      H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (((CategoryTheory.Quoti …
      X : CategoryTheory.Quotient r
      ⊢ Eq (Φ.obj X) (Φ.obj ((CategoryTheory.Quotient.functor r).obj X.as))
    -/
    congr
    /-
      🎉 no goals
    -/
    /-
      case h_map
      C : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_1, u_3} C
      r : HomRel C
      D : Type u_4
      inst✝ : CategoryTheory.Category.{u_2, u_4} D
      Φ : CategoryTheory.Functor (CategoryTheory.Quotient r) D
      H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (((CategoryTheory.Quoti …
      ⊢ ∀ (X Y : CategoryTheory.Quotient r) (f : Quiver.Hom X Y), HEq (Φ.map f) ((Ca …
    -/
  · rintro _ _ f
    /-
      case h_map
      C : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_1, u_3} C
      r : HomRel C
      D : Type u_4
      inst✝ : CategoryTheory.Category.{u_2, u_4} D
      Φ : CategoryTheory.Functor (CategoryTheory.Quotient r) D
      H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (((CategoryTheory.Quoti …
      X✝ Y✝ : CategoryTheory.Quotient r
      f : Quiver.Hom X✝ Y✝
      ⊢ HEq (Φ.map f) ((CategoryTheory.Quotient.lift r ((CategoryTheory.Quotient.fun …
    -/
    dsimp [lift, Functor]
    /-
      case h_map
      C : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_1, u_3} C
      r : HomRel C
      D : Type u_4
      inst✝ : CategoryTheory.Category.{u_2, u_4} D
      Φ : CategoryTheory.Functor (CategoryTheory.Quotient r) D
      H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (((CategoryTheory.Quoti …
      X✝ Y✝ : CategoryTheory.Quotient r
      f : Quiver.Hom X✝ Y✝
      ⊢ HEq (Φ.map f) (Quot.liftOn f (fun f => Φ.map ((CategoryTheory.Quotient.funct …
    -/
    refine Quot.inductionOn f (fun _ ↦ ?_) -- Porting note: this line was originally an `apply`
    /-
      case h_map
      C : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_1, u_3} C
      r : HomRel C
      D : Type u_4
      inst✝ : CategoryTheory.Category.{u_2, u_4} D
      Φ : CategoryTheory.Functor (CategoryTheory.Quotient r) D
      H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (((CategoryTheory.Quoti …
      X✝ Y✝ : CategoryTheory.Quotient r
      f : Quiver.Hom X✝ Y✝
      x✝ : Quiver.Hom X✝.as Y✝.as
      ⊢ HEq (Φ.map (Quot.mk (CategoryTheory.Quotient.CompClosure r) x✝)) ((Quot.mk ( …
    -/
    simp only [heq_eq_eq]
    /-
      case h_map
      C : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_1, u_3} C
      r : HomRel C
      D : Type u_4
      inst✝ : CategoryTheory.Category.{u_2, u_4} D
      Φ : CategoryTheory.Functor (CategoryTheory.Quotient r) D
      H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (((CategoryTheory.Quoti …
      X✝ Y✝ : CategoryTheory.Quotient r
      f : Quiver.Hom X✝ Y✝
      x✝ : Quiver.Hom X✝.as Y✝.as
      ⊢ Eq (Φ.map (Quot.mk (CategoryTheory.Quotient.CompClosure r) x✝)) ((Quot.mk (C …
    -/
    congr
    /-
      🎉 no goals
    -/


lemma lift_unique' (F₁ F₂ : Quotient r ⥤ D) (h : functor r ⋙ F₁ = functor r ⋙ F₂) :
    F₁ = F₂ := by
  /-
    C : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_1, u_3} C
    r : HomRel C
    D : Type u_4
    inst✝ : CategoryTheory.Category.{u_2, u_4} D
    F₁ F₂ : CategoryTheory.Functor (CategoryTheory.Quotient r) D
    h : Eq ((CategoryTheory.Quotient.functor r).comp F₁) ((CategoryTheory.Quotient …
    ⊢ Eq F₁ F₂
  -/
  rw [lift_unique r (functor r ⋙ F₂) _ F₂ rfl]; swap
    /-
      C : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_1, u_3} C
      r : HomRel C
      D : Type u_4
      inst✝ : CategoryTheory.Category.{u_2, u_4} D
      F₁ F₂ : CategoryTheory.Functor (CategoryTheory.Quotient r) D
      h : Eq ((CategoryTheory.Quotient.functor r).comp F₁) ((CategoryTheory.Quotient …
      ⊢ ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (((CategoryTheory.Quotien …
    -/
  · rintro X Y f g h
    /-
      C : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_1, u_3} C
      r : HomRel C
      D : Type u_4
      inst✝ : CategoryTheory.Category.{u_2, u_4} D
      F₁ F₂ : CategoryTheory.Functor (CategoryTheory.Quotient r) D
      h✝ : Eq ((CategoryTheory.Quotient.functor r).comp F₁) ((CategoryTheory.Quotien …
      X Y : C
      f g : Quiver.Hom X Y
      h : r f g
      ⊢ Eq (((CategoryTheory.Quotient.functor r).comp F₂).map f) (((CategoryTheory.Q …
    -/
    dsimp
    /-
      C : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_1, u_3} C
      r : HomRel C
      D : Type u_4
      inst✝ : CategoryTheory.Category.{u_2, u_4} D
      F₁ F₂ : CategoryTheory.Functor (CategoryTheory.Quotient r) D
      h✝ : Eq ((CategoryTheory.Quotient.functor r).comp F₁) ((CategoryTheory.Quotien …
      X Y : C
      f g : Quiver.Hom X Y
      h : r f g
      ⊢ Eq (F₂.map ((CategoryTheory.Quotient.functor r).map f)) (F₂.map ((CategoryTh …
    -/
    rw [Quotient.sound r h]
    /-
      🎉 no goals
    -/
  /-
    C : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_1, u_3} C
    r : HomRel C
    D : Type u_4
    inst✝ : CategoryTheory.Category.{u_2, u_4} D
    F₁ F₂ : CategoryTheory.Functor (CategoryTheory.Quotient r) D
    h : Eq ((CategoryTheory.Quotient.functor r).comp F₁) ((CategoryTheory.Quotient …
    ⊢ Eq F₁ (CategoryTheory.Quotient.lift r ((CategoryTheory.Quotient.functor r).c …
  -/
  apply lift_unique
  /-
    case hΦ
    C : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_1, u_3} C
    r : HomRel C
    D : Type u_4
    inst✝ : CategoryTheory.Category.{u_2, u_4} D
    F₁ F₂ : CategoryTheory.Functor (CategoryTheory.Quotient r) D
    h : Eq ((CategoryTheory.Quotient.functor r).comp F₁) ((CategoryTheory.Quotient …
    ⊢ Eq ((CategoryTheory.Quotient.functor r).comp F₁) ((CategoryTheory.Quotient.f …
  -/
  rw [h]
  /-
    🎉 no goals
  -/


/-- The original functor factors through the induced functor. -/
def lift.isLift : functor r ⋙ lift r F H ≅ F :=
  /-
    C : Type ?u.20598
    inst✝¹ : CategoryTheory.Category.{?u.20602, ?u.20598} C
    r : HomRel C
    D : Type ?u.20630
    inst✝ : CategoryTheory.Category.{?u.20634, ?u.20630} D
    F : CategoryTheory.Functor C D
    H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
    ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
  -/
  NatIso.ofComponents fun _ ↦ Iso.refl _
  /-
    🎉 no goals
  -/


@[simp]
theorem lift.isLift_hom (X : C) : (lift.isLift r F H).hom.app X = 𝟙 (F.obj X) :=
  rfl


@[simp]
theorem lift.isLift_inv (X : C) : (lift.isLift r F H).inv.app X = 𝟙 (F.obj X) :=
  rfl


theorem lift_obj_functor_obj (X : C) :
    (lift r F H).obj ((functor r).obj X) = F.obj X := rfl


theorem lift_map_functor_map {X Y : C} (f : X ⟶ Y) :
    (lift r F H).map ((functor r).map f) = F.map f := by
  /-
    C : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_1, u_2} C
    r : HomRel C
    D : Type u_4
    inst✝ : CategoryTheory.Category.{u_3, u_4} D
    F : CategoryTheory.Functor C D
    H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq ((CategoryTheory.Quotient.lift r F H).map ((CategoryTheory.Quotient.funct …
  -/
  rw [← NatIso.naturality_1 (lift.isLift r F H)]
  /-
    C : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_1, u_2} C
    r : HomRel C
    D : Type u_4
    inst✝ : CategoryTheory.Category.{u_3, u_4} D
    F : CategoryTheory.Functor C D
    H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq ((CategoryTheory.Quotient.lift r F H).map ((CategoryTheory.Quotient.funct …
  -/
  dsimp [lift, functor]
  /-
    C : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_1, u_2} C
    r : HomRel C
    D : Type u_4
    inst✝ : CategoryTheory.Category.{u_3, u_4} D
    F : CategoryTheory.Functor C D
    H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq ((Quot.mk (CategoryTheory.Quotient.CompClosure r) f).liftOn (fun f => F.m …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma natTrans_ext {F G : Quotient r ⥤ D} (τ₁ τ₂ : F ⟶ G)
    (h : whiskerLeft (Quotient.functor r) τ₁ = whiskerLeft (Quotient.functor r) τ₂) : τ₁ = τ₂ :=
                   /-
                     C : Type u_3
                     inst✝¹ : CategoryTheory.Category.{u_1, u_3} C
                     r : HomRel C
                     D : Type u_4
                     inst✝ : CategoryTheory.Category.{u_2, u_4} D
                     F G : CategoryTheory.Functor (CategoryTheory.Quotient r) D
                     τ₁ τ₂ : Quiver.Hom F G
                     h : Eq (CategoryTheory.whiskerLeft (CategoryTheory.Quotient.functor r) τ₁) (Ca …
                     ⊢ Eq τ₁.app τ₂.app
                   -/
  NatTrans.ext (by ext1 ⟨X⟩; exact NatTrans.congr_app h X)
                             /-
                               🎉 no goals
                             -/


/-- In order to define a natural transformation `F ⟶ G` with `F G : Quotient r ⥤ D`, it suffices
to do so after precomposing with `Quotient.functor r`. -/
def natTransLift {F G : Quotient r ⥤ D} (τ : Quotient.functor r ⋙ F ⟶ Quotient.functor r ⋙ G) :
    F ⟶ G where
  app := fun ⟨X⟩ => τ.app X
  naturality := fun ⟨X⟩ ⟨Y⟩ => by
    /-
      C : Type ?u.25354
      inst✝¹ : CategoryTheory.Category.{?u.25358, ?u.25354} C
      r : HomRel C
      D : Type ?u.25386
      inst✝ : CategoryTheory.Category.{?u.25390, ?u.25386} D
      F✝ : CategoryTheory.Functor C D
      H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F✝.map f₁) (F✝.map f₂)
      F G : CategoryTheory.Functor (CategoryTheory.Quotient r) D
      τ : Quiver.Hom ((CategoryTheory.Quotient.functor r).comp F) ((CategoryTheory.Q …
      x✝¹ x✝ : CategoryTheory.Quotient r
      X Y : C
      ⊢ ∀ (f : Quiver.Hom { as := X } { as := Y }), Eq (CategoryTheory.CategoryStruc …
    -/
    rintro ⟨f⟩
    /-
      case mk
      C : Type ?u.25354
      inst✝¹ : CategoryTheory.Category.{?u.25358, ?u.25354} C
      r : HomRel C
      D : Type ?u.25386
      inst✝ : CategoryTheory.Category.{?u.25390, ?u.25386} D
      F✝ : CategoryTheory.Functor C D
      H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F✝.map f₁) (F✝.map f₂)
      F G : CategoryTheory.Functor (CategoryTheory.Quotient r) D
      τ : Quiver.Hom ((CategoryTheory.Quotient.functor r).comp F) ((CategoryTheory.Q …
      x✝¹ x✝ : CategoryTheory.Quotient r
      X Y : C
      f✝ : Quiver.Hom { as := X } { as := Y }
      f : Quiver.Hom { as := X }.as { as := Y }.as
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (Quot.mk (CategoryTheory.Quoti …
    -/
    exact τ.naturality f
    /-
      🎉 no goals
    -/


@[simp]
lemma natTransLift_app (F G : Quotient r ⥤ D)
    (τ : Quotient.functor r ⋙ F ⟶ Quotient.functor r ⋙ G) (X : C) :
  (natTransLift r τ).app ((Quotient.functor r).obj X) = τ.app X := rfl


@[reassoc]
lemma comp_natTransLift {F G H : Quotient r ⥤ D}
    (τ : Quotient.functor r ⋙ F ⟶ Quotient.functor r ⋙ G)
    (τ' : Quotient.functor r ⋙ G ⟶ Quotient.functor r ⋙ H) :
                                                                          /-
                                                                            C : Type u_3
                                                                            inst✝¹ : CategoryTheory.Category.{u_1, u_3} C
                                                                            r : HomRel C
                                                                            D : Type u_4
                                                                            inst✝ : CategoryTheory.Category.{u_2, u_4} D
                                                                            F G H : CategoryTheory.Functor (CategoryTheory.Quotient r) D
                                                                            τ : Quiver.Hom ((CategoryTheory.Quotient.functor r).comp F) ((CategoryTheory.Q …
                                                                            τ' : Quiver.Hom ((CategoryTheory.Quotient.functor r).comp G) ((CategoryTheory. …
                                                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Quotient.natTransLift …
                                                                          -/
    natTransLift r τ ≫ natTransLift r τ' =  natTransLift r (τ ≫ τ') := by aesop_cat
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp]
lemma natTransLift_id (F : Quotient r ⥤ D) :
                                                            /-
                                                              C : Type u_3
                                                              inst✝¹ : CategoryTheory.Category.{u_1, u_3} C
                                                              r : HomRel C
                                                              D : Type u_4
                                                              inst✝ : CategoryTheory.Category.{u_2, u_4} D
                                                              F : CategoryTheory.Functor (CategoryTheory.Quotient r) D
                                                              ⊢ Eq (CategoryTheory.Quotient.natTransLift r (CategoryTheory.CategoryStruct.id …
                                                            -/
    natTransLift r (𝟙 (Quotient.functor r ⋙ F)) = 𝟙 _ := by aesop_cat
                                                            /-
                                                              🎉 no goals
                                                            -/


/-- In order to define a natural isomorphism `F ≅ G` with `F G : Quotient r ⥤ D`, it suffices
to do so after precomposing with `Quotient.functor r`. -/
@[simps]
def natIsoLift {F G : Quotient r ⥤ D} (τ : Quotient.functor r ⋙ F ≅ Quotient.functor r ⋙ G) :
    F ≅ G where
  hom := natTransLift _ τ.hom
  inv := natTransLift _ τ.inv
                   /-
                     C : Type ?u.29973
                     inst✝¹ : CategoryTheory.Category.{?u.29977, ?u.29973} C
                     r : HomRel C
                     D : Type ?u.30005
                     inst✝ : CategoryTheory.Category.{?u.30009, ?u.30005} D
                     F✝ : CategoryTheory.Functor C D
                     H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F✝.map f₁) (F✝.map f₂)
                     F G : CategoryTheory.Functor (CategoryTheory.Quotient r) D
                     τ : CategoryTheory.Iso ((CategoryTheory.Quotient.functor r).comp F) ((Category …
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Quotient.natTransLift …
                   -/
  hom_inv_id := by rw [comp_natTransLift, τ.hom_inv_id, natTransLift_id]
                   /-
                     🎉 no goals
                   -/
                   /-
                     C : Type ?u.29973
                     inst✝¹ : CategoryTheory.Category.{?u.29977, ?u.29973} C
                     r : HomRel C
                     D : Type ?u.30005
                     inst✝ : CategoryTheory.Category.{?u.30009, ?u.30005} D
                     F✝ : CategoryTheory.Functor C D
                     H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F✝.map f₁) (F✝.map f₂)
                     F G : CategoryTheory.Functor (CategoryTheory.Quotient r) D
                     τ : CategoryTheory.Iso ((CategoryTheory.Quotient.functor r).comp F) ((Category …
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Quotient.natTransLift …
                   -/
  inv_hom_id := by rw [comp_natTransLift, τ.inv_hom_id, natTransLift_id]
                   /-
                     🎉 no goals
                   -/


instance full_whiskeringLeft_functor :
    ((whiskeringLeft C _ D).obj (functor r)).Full where
                                            /-
                                              C : Type u_1
                                              inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
                                              r : HomRel C
                                              D : Type u_4
                                              inst✝ : CategoryTheory.Category.{u_2, u_4} D
                                              F : CategoryTheory.Functor C D
                                              H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
                                              X✝ Y✝ : CategoryTheory.Functor (CategoryTheory.Quotient r) D
                                              f : Quiver.Hom (((CategoryTheory.whiskeringLeft C (CategoryTheory.Quotient r)  …
                                              ⊢ Eq (((CategoryTheory.whiskeringLeft C (CategoryTheory.Quotient r) D).obj (Ca …
                                            -/
  map_surjective f := ⟨natTransLift r f, by aesop_cat⟩
                                            /-
                                              🎉 no goals
                                            -/


instance faithful_whiskeringLeft_functor :
                                                             /-
                                                               C : Type u_1
                                                               inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
                                                               r : HomRel C
                                                               D : Type u_4
                                                               inst✝ : CategoryTheory.Category.{u_2, u_4} D
                                                               F : CategoryTheory.Functor C D
                                                               H : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
                                                               ⊢ ∀ {X Y : CategoryTheory.Functor (CategoryTheory.Quotient r) D}, Function.Inj …
                                                             -/
    ((whiskeringLeft C _ D).obj (functor r)).Faithful := ⟨by apply natTrans_ext⟩
                                                             /-
                                                               🎉 no goals
                                                             -/


