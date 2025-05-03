/-- In a preadditive category `C`, when an object `X` decomposes as `X ≅ P ⨿ Q`, one may
consider `P` as a direct factor of `X` and up to unique isomorphism, it is determined by the
obvious idempotent `X ⟶ P ⟶ X` which is the projection onto `P` with kernel `Q`. More generally,
one may define a formal direct factor of an object `X : C` : it consists of an idempotent
`p : X ⟶ X` which is thought as the "formal image" of `p`. The type `Karoubi C` shall be the
type of the objects of the karoubi envelope of `C`. It makes sense for any category `C`. -/
structure Karoubi where
  /-- an object of the underlying category -/
  X : C
  /-- an endomorphism of the object -/
  p : X ⟶ X
  /-- the condition that the given endomorphism is an idempotent -/
  idem : p ≫ p = p := by aesop_cat


attribute [reassoc (attr := simp)] idem


@[ext (iff := false)]
theorem ext {P Q : Karoubi C} (h_X : P.X = Q.X) (h_p : P.p ≫ eqToHom h_X = eqToHom h_X ≫ Q.p) :
    P = Q := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    P Q : CategoryTheory.Idempotents.Karoubi C
    h_X : Eq P.X Q.X
    h_p : Eq (CategoryTheory.CategoryStruct.comp P.p (CategoryTheory.eqToHom h_X)) …
    ⊢ Eq P Q
  -/
  cases P
  /-
    case mk
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    Q : CategoryTheory.Idempotents.Karoubi C
    X✝ : C
    p✝ : Quiver.Hom X✝ X✝
    idem✝ : Eq (CategoryTheory.CategoryStruct.comp p✝ p✝) p✝
    h_X : Eq { X := X✝, p := p✝, idem := idem✝ }.X Q.X
    h_p : Eq (CategoryTheory.CategoryStruct.comp { X := X✝, p := p✝, idem := idem✝ …
    ⊢ Eq { X := X✝, p := p✝, idem := idem✝ } Q
  -/
  cases Q
  /-
    case mk.mk
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X✝¹ : C
    p✝¹ : Quiver.Hom X✝¹ X✝¹
    idem✝¹ : Eq (CategoryTheory.CategoryStruct.comp p✝¹ p✝¹) p✝¹
    X✝ : C
    p✝ : Quiver.Hom X✝ X✝
    idem✝ : Eq (CategoryTheory.CategoryStruct.comp p✝ p✝) p✝
    h_X : Eq { X := X✝¹, p := p✝¹, idem := idem✝¹ }.X { X := X✝, p := p✝, idem :=  …
    h_p : Eq (CategoryTheory.CategoryStruct.comp { X := X✝¹, p := p✝¹, idem := ide …
    ⊢ Eq { X := X✝¹, p := p✝¹, idem := idem✝¹ } { X := X✝, p := p✝, idem := idem✝ }
  -/
  dsimp at h_X h_p
  /-
    case mk.mk
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X✝¹ : C
    p✝¹ : Quiver.Hom X✝¹ X✝¹
    idem✝¹ : Eq (CategoryTheory.CategoryStruct.comp p✝¹ p✝¹) p✝¹
    X✝ : C
    p✝ : Quiver.Hom X✝ X✝
    idem✝ : Eq (CategoryTheory.CategoryStruct.comp p✝ p✝) p✝
    h_X : Eq X✝¹ X✝
    h_p : Eq (CategoryTheory.CategoryStruct.comp p✝¹ (CategoryTheory.eqToHom h_X)) …
    ⊢ Eq { X := X✝¹, p := p✝¹, idem := idem✝¹ } { X := X✝, p := p✝, idem := idem✝ }
  -/
  subst h_X
  /-
    case mk.mk
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X✝ : C
    p✝¹ : Quiver.Hom X✝ X✝
    idem✝¹ : Eq (CategoryTheory.CategoryStruct.comp p✝¹ p✝¹) p✝¹
    p✝ : Quiver.Hom X✝ X✝
    idem✝ : Eq (CategoryTheory.CategoryStruct.comp p✝ p✝) p✝
    h_p : Eq (CategoryTheory.CategoryStruct.comp p✝¹ (CategoryTheory.eqToHom ⋯)) ( …
    ⊢ Eq { X := X✝, p := p✝¹, idem := idem✝¹ } { X := X✝, p := p✝, idem := idem✝ }
  -/
  simpa only [mk.injEq, heq_eq_eq, true_and, eqToHom_refl, comp_id, id_comp] using h_p
  /-
    🎉 no goals
  -/


/-- A morphism `P ⟶ Q` in the category `Karoubi C` is a morphism in the underlying category
`C` which satisfies a relation, which in the preadditive case, expresses that it induces a
map between the corresponding "formal direct factors" and that it vanishes on the complement
formal direct factor. -/
@[ext]
structure Hom (P Q : Karoubi C) where
  /-- a morphism between the underlying objects -/
  f : P.X ⟶ Q.X
  /-- compatibility of the given morphism with the given idempotents -/
  comm : f = P.p ≫ f ≫ Q.p := by aesop_cat


instance [Preadditive C] (P Q : Karoubi C) : Inhabited (Hom P Q) :=
          /-
            C : Type u_1
            inst✝¹ : CategoryTheory.Category.{?u.3411, u_1} C
            inst✝ : CategoryTheory.Preadditive C
            P Q : CategoryTheory.Idempotents.Karoubi C
            ⊢ Eq 0 (CategoryTheory.CategoryStruct.comp P.p (CategoryTheory.CategoryStruct. …
          -/
  ⟨⟨0, by rw [zero_comp, comp_zero]⟩⟩
          /-
            🎉 no goals
          -/


@[reassoc (attr := simp)]
                                                                       /-
                                                                         C : Type u_1
                                                                         inst✝ : CategoryTheory.Category.{u_2, u_1} C
                                                                         P Q : CategoryTheory.Idempotents.Karoubi C
                                                                         f : P.Hom Q
                                                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp P.p f.f) f.f
                                                                       -/
theorem p_comp {P Q : Karoubi C} (f : Hom P Q) : P.p ≫ f.f = f.f := by rw [f.comm, ← assoc, P.idem]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[reassoc (attr := simp)]
theorem comp_p {P Q : Karoubi C} (f : Hom P Q) : f.f ≫ Q.p = f.f := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    P Q : CategoryTheory.Idempotents.Karoubi C
    f : P.Hom Q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f.f Q.p) f.f
  -/
  rw [f.comm, assoc, assoc, Q.idem]
  /-
    🎉 no goals
  -/


@[reassoc]
                                                                             /-
                                                                               C : Type u_1
                                                                               inst✝ : CategoryTheory.Category.{u_2, u_1} C
                                                                               P Q : CategoryTheory.Idempotents.Karoubi C
                                                                               f : P.Hom Q
                                                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp P.p f.f) (CategoryTheory.CategoryStru …
                                                                             -/
theorem p_comm {P Q : Karoubi C} (f : Hom P Q) : P.p ≫ f.f = f.f ≫ Q.p := by rw [p_comp, comp_p]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem comp_proof {P Q R : Karoubi C} (g : Hom Q R) (f : Hom P Q) :
                                              /-
                                                C : Type u_1
                                                inst✝ : CategoryTheory.Category.{u_2, u_1} C
                                                P Q R : CategoryTheory.Idempotents.Karoubi C
                                                g : Q.Hom R
                                                f : P.Hom Q
                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp f.f g.f) (CategoryTheory.CategoryStru …
                                              -/
    f.f ≫ g.f = P.p ≫ (f.f ≫ g.f) ≫ R.p := by rw [assoc, comp_p, ← assoc, p_comp]
                                              /-
                                                🎉 no goals
                                              -/


/-- The category structure on the karoubi envelope of a category. -/
instance : Category (Karoubi C) where
  Hom := Karoubi.Hom
                   /-
                     C : Type u_1
                     inst✝ : CategoryTheory.Category.{?u.12290, u_1} C
                     P : CategoryTheory.Idempotents.Karoubi C
                     ⊢ Eq P.p (CategoryTheory.CategoryStruct.comp P.p (CategoryTheory.CategoryStruc …
                   -/
  id P := ⟨P.p, by repeat' rw [P.idem]⟩
                   /-
                     🎉 no goals
                   -/
  comp f g := ⟨f.f ≫ g.f, Karoubi.comp_proof g f⟩


@[simp]
theorem hom_ext_iff {P Q : Karoubi C} {f g : P ⟶ Q} : f = g ↔ f.f = g.f := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    P Q : CategoryTheory.Idempotents.Karoubi C
    f g : Quiver.Hom P Q
    ⊢ Iff (Eq f g) (Eq f.f g.f)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      P Q : CategoryTheory.Idempotents.Karoubi C
      f g : Quiver.Hom P Q
      ⊢ Eq f g → Eq f.f g.f
    -/
  · intro h
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      P Q : CategoryTheory.Idempotents.Karoubi C
      f g : Quiver.Hom P Q
      h : Eq f g
      ⊢ Eq f.f g.f
    -/
    rw [h]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      P Q : CategoryTheory.Idempotents.Karoubi C
      f g : Quiver.Hom P Q
      ⊢ Eq f.f g.f → Eq f g
    -/
  · apply Hom.ext
    /-
      🎉 no goals
    -/


@[ext]
theorem hom_ext {P Q : Karoubi C} (f g : P ⟶ Q) (h : f.f = g.f) : f = g := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    P Q : CategoryTheory.Idempotents.Karoubi C
    f g : Quiver.Hom P Q
    h : Eq f.f g.f
    ⊢ Eq f g
  -/
  simpa [hom_ext_iff] using h
  /-
    🎉 no goals
  -/


@[simp]
theorem comp_f {P Q R : Karoubi C} (f : P ⟶ Q) (g : Q ⟶ R) : (f ≫ g).f = f.f ≫ g.f := rfl


@[simp]
theorem id_f {P : Karoubi C} : Hom.f (𝟙 P) = P.p := rfl


@[deprecated "No deprecation message was provided." (since := "2024-07-15")]
                                               /-
                                                 C : Type u_1
                                                 inst✝ : CategoryTheory.Category.{?u.15526, u_1} C
                                                 P : CategoryTheory.Idempotents.Karoubi C
                                                 ⊢ Eq P.p (CategoryTheory.CategoryStruct.comp P.p (CategoryTheory.CategoryStruc …
                                               -/
theorem id_eq {P : Karoubi C} : 𝟙 P = ⟨P.p, by repeat' rw [P.idem]⟩ := rfl
                                               /-
                                                 🎉 no goals
                                               -/


/-- It is possible to coerce an object of `C` into an object of `Karoubi C`.
See also the functor `toKaroubi`. -/
instance coe : CoeTC C (Karoubi C) :=
                        /-
                          C : Type u_1
                          inst✝ : CategoryTheory.Category.{?u.15820, u_1} C
                          X : C
                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X)  …
                        -/
  ⟨fun X => ⟨X, 𝟙 X, by rw [comp_id]⟩⟩
                        /-
                          🎉 no goals
                        -/

-- Porting note: removed @[simp] as the linter complains

theorem coe_X (X : C) : (X : Karoubi C).X = X := rfl


@[simp]
theorem coe_p (X : C) : (X : Karoubi C).p = 𝟙 X := rfl


@[simp]
theorem eqToHom_f {P Q : Karoubi C} (h : P = Q) :
    Karoubi.Hom.f (eqToHom h) = P.p ≫ eqToHom (congr_arg Karoubi.X h) := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    P Q : CategoryTheory.Idempotents.Karoubi C
    h : Eq P Q
    ⊢ Eq (CategoryTheory.eqToHom h).f (CategoryTheory.CategoryStruct.comp P.p (Cat …
  -/
  subst h
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    P : CategoryTheory.Idempotents.Karoubi C
    ⊢ Eq (CategoryTheory.eqToHom ⋯).f (CategoryTheory.CategoryStruct.comp P.p (Cat …
  -/
  simp only [eqToHom_refl, Karoubi.id_f, comp_id]
  /-
    🎉 no goals
  -/


/-- The obvious fully faithful functor `toKaroubi` sends an object `X : C` to the obvious
formal direct factor of `X` given by `𝟙 X`. -/
@[simps]
def toKaroubi : C ⥤ Karoubi C where
                       /-
                         C : Type u_1
                         inst✝ : CategoryTheory.Category.{?u.18071, u_1} C
                         X : C
                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X)  …
                       -/
  obj X := ⟨X, 𝟙 X, by rw [comp_id]⟩
                       /-
                         🎉 no goals
                       -/
                  /-
                    C : Type u_1
                    inst✝ : CategoryTheory.Category.{?u.18071, u_1} C
                    X✝ Y✝ : C
                    f : Quiver.Hom X✝ Y✝
                    ⊢ Eq f (CategoryTheory.CategoryStruct.comp ((fun X => { X := X, p := CategoryT …
                  -/
  map f := ⟨f, by simp only [comp_id, id_comp]⟩
                  /-
                    🎉 no goals
                  -/


instance : (toKaroubi C).Full where map_surjective f := ⟨f.f, rfl⟩


instance : (toKaroubi C).Faithful where
  map_injective := fun h => congr_arg Karoubi.Hom.f h


@[simps add]
instance instAdd [Preadditive C] {P Q : Karoubi C} : Add (P ⟶ Q) where
                            /-
                              C : Type u_1
                              inst✝¹ : CategoryTheory.Category.{?u.20993, u_1} C
                              inst✝ : CategoryTheory.Preadditive C
                              P Q : CategoryTheory.Idempotents.Karoubi C
                              f g : Quiver.Hom P Q
                              ⊢ Eq (HAdd.hAdd f.f g.f) (CategoryTheory.CategoryStruct.comp P.p (CategoryTheo …
                            -/
  add f g := ⟨f.f + g.f, by rw [add_comp, comp_add, ← f.comm, ← g.comm]⟩
                            /-
                              🎉 no goals
                            -/


@[simps neg]
instance instNeg [Preadditive C] {P Q : Karoubi C} : Neg (P ⟶ Q) where
                     /-
                       C : Type u_1
                       inst✝¹ : CategoryTheory.Category.{?u.22316, u_1} C
                       inst✝ : CategoryTheory.Preadditive C
                       P Q : CategoryTheory.Idempotents.Karoubi C
                       f : Quiver.Hom P Q
                       ⊢ Eq (Neg.neg f.f) (CategoryTheory.CategoryStruct.comp P.p (CategoryTheory.Cat …
                     -/
  neg f := ⟨-f.f, by simpa only [neg_comp, comp_neg, neg_inj] using f.comm⟩
                     /-
                       🎉 no goals
                     -/


@[simps zero]
instance instZero [Preadditive C] {P Q : Karoubi C} : Zero (P ⟶ Q) where
                 /-
                   C : Type u_1
                   inst✝¹ : CategoryTheory.Category.{?u.23167, u_1} C
                   inst✝ : CategoryTheory.Preadditive C
                   P Q : CategoryTheory.Idempotents.Karoubi C
                   ⊢ Eq 0 (CategoryTheory.CategoryStruct.comp P.p (CategoryTheory.CategoryStruct. …
                 -/
  zero := ⟨0, by simp only [comp_zero, zero_comp]⟩
                 /-
                   🎉 no goals
                 -/

-- dsimp loops when applying this lemma to its LHS,
-- probably https://github.com/leanprover/lean4/pull/2867

instance instAddCommGroupHom [Preadditive C] {P Q : Karoubi C} : AddCommGroup (P ⟶ Q) where
  zero_add f := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.24354, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      P Q : CategoryTheory.Idempotents.Karoubi C
      f : Quiver.Hom P Q
      ⊢ Eq (HAdd.hAdd 0 f) f
    -/
    ext
    /-
      case h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.24354, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      P Q : CategoryTheory.Idempotents.Karoubi C
      f : Quiver.Hom P Q
      ⊢ Eq (HAdd.hAdd 0 f).f f.f
    -/
    apply zero_add
    /-
      🎉 no goals
    -/
  add_zero f := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.24354, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      P Q : CategoryTheory.Idempotents.Karoubi C
      f g h' : Quiver.Hom P Q
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd f g) h') (HAdd.hAdd f (HAdd.hAdd g h'))
    -/
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.24354, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      P Q : CategoryTheory.Idempotents.Karoubi C
      f : Quiver.Hom P Q
      ⊢ Eq (HAdd.hAdd f 0) f
    -/
    /-
      case h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.24354, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      P Q : CategoryTheory.Idempotents.Karoubi C
      f g h' : Quiver.Hom P Q
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd f g) h').f (HAdd.hAdd f (HAdd.hAdd g h')).f
    -/
    ext
    /-
      🎉 no goals
    -/
    /-
      case h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.24354, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      P Q : CategoryTheory.Idempotents.Karoubi C
      f : Quiver.Hom P Q
      ⊢ Eq (HAdd.hAdd f 0).f f.f
    -/
    apply add_zero
    /-
      🎉 no goals
    -/
  add_assoc f g h' := by
    ext
    apply add_assoc
  add_comm f g := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.24354, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      P Q : CategoryTheory.Idempotents.Karoubi C
      f g : Quiver.Hom P Q
      ⊢ Eq (HAdd.hAdd f g) (HAdd.hAdd g f)
    -/
    ext
    /-
      case h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.24354, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      P Q : CategoryTheory.Idempotents.Karoubi C
      f g : Quiver.Hom P Q
      ⊢ Eq (HAdd.hAdd f g).f (HAdd.hAdd g f).f
    -/
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.24354, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      P Q : CategoryTheory.Idempotents.Karoubi C
      f : Quiver.Hom P Q
      ⊢ Eq (HAdd.hAdd (Neg.neg f) f) 0
    -/
    apply add_comm
    /-
      case h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.24354, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      P Q : CategoryTheory.Idempotents.Karoubi C
      f : Quiver.Hom P Q
      ⊢ Eq (HAdd.hAdd (Neg.neg f) f).f (CategoryTheory.Idempotents.Karoubi.Hom.f 0)
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  neg_add_cancel f := by
    ext
    apply neg_add_cancel
  zsmul := zsmulRec
  nsmul := nsmulRec


theorem hom_eq_zero_iff [Preadditive C] {P Q : Karoubi C} {f : P ⟶ Q} : f = 0 ↔ f.f = 0 :=
  hom_ext_iff


/-- The map sending `f : P ⟶ Q` to `f.f : P.X ⟶ Q.X` is additive. -/
@[simps]
def inclusionHom [Preadditive C] (P Q : Karoubi C) : AddMonoidHom (P ⟶ Q) (P.X ⟶ Q.X) where
  toFun f := f.f
  map_zero' := rfl
  map_add' _ _ := rfl


@[simp]
theorem sum_hom [Preadditive C] {P Q : Karoubi C} {α : Type*} (s : Finset α) (f : α → (P ⟶ Q)) :
    (∑ x ∈ s, f x).f = ∑ x ∈ s, (f x).f :=
  map_sum (inclusionHom P Q) f s


/-- The category `Karoubi C` is preadditive if `C` is. -/
instance [Preadditive C] : Preadditive (Karoubi C) where
                     /-
                       C : Type u_1
                       inst✝¹ : CategoryTheory.Category.{?u.28688, u_1} C
                       inst✝ : CategoryTheory.Preadditive C
                       P Q : CategoryTheory.Idempotents.Karoubi C
                       ⊢ AddCommGroup (Quiver.Hom P Q)
                     -/
  homGroup P Q := by infer_instance
                     /-
                       🎉 no goals
                     -/


instance [Preadditive C] : Functor.Additive (toKaroubi C) where


instance : IsIdempotentComplete (Karoubi C) := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    ⊢ CategoryTheory.IsIdempotentComplete (CategoryTheory.Idempotents.Karoubi C)
  -/
  refine ⟨?_⟩
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    ⊢ ∀ (X : CategoryTheory.Idempotents.Karoubi C) (p : Quiver.Hom X X), Eq (Categ …
  -/
  intro P p hp
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    P : CategoryTheory.Idempotents.Karoubi C
    p : Quiver.Hom P P
    hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
    ⊢ Exists fun Y => Exists fun i => Exists fun e => And (Eq (CategoryTheory.Cate …
  -/
  simp only [hom_ext_iff, comp_f] at hp
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    P : CategoryTheory.Idempotents.Karoubi C
    p : Quiver.Hom P P
    hp : Eq (CategoryTheory.CategoryStruct.comp p.f p.f) p.f
    ⊢ Exists fun Y => Exists fun i => Exists fun e => And (Eq (CategoryTheory.Cate …
  -/
  use ⟨P.X, p.f, hp⟩
  /-
    case h
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    P : CategoryTheory.Idempotents.Karoubi C
    p : Quiver.Hom P P
    hp : Eq (CategoryTheory.CategoryStruct.comp p.f p.f) p.f
    ⊢ Exists fun i => Exists fun e => And (Eq (CategoryTheory.CategoryStruct.comp  …
  -/
  use ⟨p.f, by rw [comp_p p, hp]⟩
  /-
    case h
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    P : CategoryTheory.Idempotents.Karoubi C
    p : Quiver.Hom P P
    hp : Eq (CategoryTheory.CategoryStruct.comp p.f p.f) p.f
    ⊢ Exists fun e => And (Eq (CategoryTheory.CategoryStruct.comp { f := p.f, comm …
  -/
  use ⟨p.f, by rw [hp, p_comp p]⟩
  /-
    case h
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    P : CategoryTheory.Idempotents.Karoubi C
    p : Quiver.Hom P P
    hp : Eq (CategoryTheory.CategoryStruct.comp p.f p.f) p.f
    ⊢ And (Eq (CategoryTheory.CategoryStruct.comp { f := p.f, comm := ⋯ } { f := p …
  -/
  simp [hp]
  /-
    🎉 no goals
  -/


instance [IsIdempotentComplete C] : (toKaroubi C).EssSurj :=
  ⟨fun P => by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.IsIdempotentComplete C
      P : CategoryTheory.Idempotents.Karoubi C
      ⊢ Membership.mem (CategoryTheory.Idempotents.toKaroubi C).essImage P
    -/
    rcases IsIdempotentComplete.idempotents_split P.X P.p P.idem with ⟨Y, i, e, ⟨h₁, h₂⟩⟩
    /-
      case intro.intro.intro.intro
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.IsIdempotentComplete C
      P : CategoryTheory.Idempotents.Karoubi C
      Y : C
      i : Quiver.Hom Y P.X
      e : Quiver.Hom P.X Y
      h₁ : Eq (CategoryTheory.CategoryStruct.comp i e) (CategoryTheory.CategoryStruc …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp e i) P.p
      ⊢ Membership.mem (CategoryTheory.Idempotents.toKaroubi C).essImage P
    -/
    use Y
    exact
      Nonempty.intro
        { hom := ⟨i, by erw [id_comp, ← h₂, ← assoc, h₁, id_comp]⟩
          inv := ⟨e, by erw [comp_id, ← h₂, assoc, h₁, comp_id]⟩ }⟩


/-- If `C` is idempotent complete, the functor `toKaroubi : C ⥤ Karoubi C` is an equivalence. -/
instance toKaroubi_isEquivalence [IsIdempotentComplete C] : (toKaroubi C).IsEquivalence where


/-- The equivalence `C ≅ Karoubi C` when `C` is idempotent complete. -/
def toKaroubiEquivalence [IsIdempotentComplete C] : C ≌ Karoubi C :=
  (toKaroubi C).asEquivalence


instance toKaroubiEquivalence_functor_additive [Preadditive C] [IsIdempotentComplete C] :
    (toKaroubiEquivalence C).functor.Additive :=
  (inferInstance : (toKaroubi C).Additive)


/-- The split mono which appears in the factorisation `decompId P`. -/
@[simps]
def decompId_i (P : Karoubi C) : P ⟶ P.X :=
           /-
             C : Type u_1
             inst✝ : CategoryTheory.Category.{?u.37048, u_1} C
             P : CategoryTheory.Idempotents.Karoubi C
             ⊢ Eq P.p (CategoryTheory.CategoryStruct.comp P.p (CategoryTheory.CategoryStruc …
           -/
  ⟨P.p, by rw [coe_p, comp_id, P.idem]⟩
           /-
             🎉 no goals
           -/


/-- The split epi which appears in the factorisation `decompId P`. -/
@[simps]
def decompId_p (P : Karoubi C) : (P.X : Karoubi C) ⟶ P :=
           /-
             C : Type u_1
             inst✝ : CategoryTheory.Category.{?u.38658, u_1} C
             P : CategoryTheory.Idempotents.Karoubi C
             ⊢ Eq P.p (CategoryTheory.CategoryStruct.comp { X := P.X, p := CategoryTheory.C …
           -/
  ⟨P.p, by rw [coe_p, id_comp, P.idem]⟩
           /-
             🎉 no goals
           -/


/-- The formal direct factor of `P.X` given by the idempotent `P.p` in the category `C`
is actually a direct factor in the category `Karoubi C`. -/
@[reassoc]
theorem decompId (P : Karoubi C) : 𝟙 P = decompId_i P ≫ decompId_p P := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    P : CategoryTheory.Idempotents.Karoubi C
    ⊢ Eq (CategoryTheory.CategoryStruct.id P) (CategoryTheory.CategoryStruct.comp  …
  -/
  ext
  /-
    case h
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    P : CategoryTheory.Idempotents.Karoubi C
    ⊢ Eq (CategoryTheory.CategoryStruct.id P).f (CategoryTheory.CategoryStruct.com …
  -/
  simp only [comp_f, id_f, P.idem, decompId_i, decompId_p]
  /-
    🎉 no goals
  -/


theorem decomp_p (P : Karoubi C) : (toKaroubi C).map P.p = decompId_p P ≫ decompId_i P := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    P : CategoryTheory.Idempotents.Karoubi C
    ⊢ Eq ((CategoryTheory.Idempotents.toKaroubi C).map P.p) (CategoryTheory.Catego …
  -/
  ext
  /-
    case h
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    P : CategoryTheory.Idempotents.Karoubi C
    ⊢ Eq ((CategoryTheory.Idempotents.toKaroubi C).map P.p).f (CategoryTheory.Cate …
  -/
  simp only [comp_f, decompId_p_f, decompId_i_f, P.idem, toKaroubi_map_f]
  /-
    🎉 no goals
  -/


theorem decompId_i_toKaroubi (X : C) : decompId_i ((toKaroubi C).obj X) = 𝟙 _ := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X : C
    ⊢ Eq ((CategoryTheory.Idempotents.toKaroubi C).obj X).decompId_i (CategoryTheo …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem decompId_p_toKaroubi (X : C) : decompId_p ((toKaroubi C).obj X) = 𝟙 _ := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X : C
    ⊢ Eq ((CategoryTheory.Idempotents.toKaroubi C).obj X).decompId_p (CategoryTheo …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem decompId_i_naturality {P Q : Karoubi C} (f : P ⟶ Q) :
                                          /-
                                            C : Type u_1
                                            inst✝ : CategoryTheory.Category.{?u.41746, u_1} C
                                            P Q : CategoryTheory.Idempotents.Karoubi C
                                            f : Quiver.Hom P Q
                                            ⊢ Quiver.Hom { X := P.X, p := CategoryTheory.CategoryStruct.id P.X, idem := ⋯  …
                                          -/
    f ≫ decompId_i Q = decompId_i P ≫ (by exact Hom.mk f.f (by simp)) := by
                                          /-
                                            🎉 no goals
                                          -/
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    P Q : CategoryTheory.Idempotents.Karoubi C
    f : Quiver.Hom P Q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f Q.decompId_i) (CategoryTheory.Categ …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


theorem decompId_p_naturality {P Q : Karoubi C} (f : P ⟶ Q) :
                           /-
                             C : Type u_1
                             inst✝ : CategoryTheory.Category.{?u.42574, u_1} C
                             P Q : CategoryTheory.Idempotents.Karoubi C
                             f : Quiver.Hom P Q
                             ⊢ Quiver.Hom { X := P.X, p := CategoryTheory.CategoryStruct.id P.X, idem := ⋯  …
                           -/
    decompId_p P ≫ f = (by exact Hom.mk f.f (by simp)) ≫ decompId_p Q := by
                           /-
                             🎉 no goals
                           -/
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    P Q : CategoryTheory.Idempotents.Karoubi C
    f : Quiver.Hom P Q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp P.decompId_p f) (CategoryTheory.Categ …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


@[simp]
theorem zsmul_hom [Preadditive C] {P Q : Karoubi C} (n : ℤ) (f : P ⟶ Q) : (n • f).f = n • f.f :=
  map_zsmul (inclusionHom P Q) n f


