/-- A `MorphismProperty C` is a class of morphisms between objects in `C`. -/
def MorphismProperty :=
  ∀ ⦃X Y : C⦄ (_ : X ⟶ Y), Prop


instance : CompleteBooleanAlgebra (MorphismProperty C) where
  le P₁ P₂ := ∀ ⦃X Y : C⦄ (f : X ⟶ Y), P₁ f → P₂ f
  __ := inferInstanceAs (CompleteBooleanAlgebra (∀ ⦃X Y : C⦄ (_ : X ⟶ Y), Prop))


lemma MorphismProperty.le_def {P Q : MorphismProperty C} :
    P ≤ Q ↔ ∀ {X Y : C} (f : X ⟶ Y), P f → Q f := Iff.rfl


instance : Inhabited (MorphismProperty C) :=
  ⟨⊤⟩


lemma MorphismProperty.top_eq : (⊤ : MorphismProperty C) = fun _ _ _ => True := rfl


@[ext]
lemma ext (W W' : MorphismProperty C) (h : ∀ ⦃X Y : C⦄ (f : X ⟶ Y), W f ↔ W' f) :
    W = W' := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    W W' : CategoryTheory.MorphismProperty C
    h : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), Iff (W f) (W' f)
    ⊢ Eq W W'
  -/
  funext X Y f
  /-
    case h.h.h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    W W' : CategoryTheory.MorphismProperty C
    h : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), Iff (W f) (W' f)
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq (W f) (W' f)
  -/
  rw [h]
  /-
    🎉 no goals
  -/


@[simp]
lemma top_apply {X Y : C} (f : X ⟶ Y) : (⊤ : MorphismProperty C) f := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Top.top f
  -/
  simp only [top_eq]
  /-
    🎉 no goals
  -/


/-- The morphism property in `Cᵒᵖ` associated to a morphism property in `C` -/
@[simp]
def op (P : MorphismProperty C) : MorphismProperty Cᵒᵖ := fun _ _ f => P f.unop


/-- The morphism property in `C` associated to a morphism property in `Cᵒᵖ` -/
@[simp]
def unop (P : MorphismProperty Cᵒᵖ) : MorphismProperty C := fun _ _ f => P f.op


theorem unop_op (P : MorphismProperty C) : P.op.unop = P :=
  rfl


theorem op_unop (P : MorphismProperty Cᵒᵖ) : P.unop.op = P :=
  rfl


/-- The inverse image of a `MorphismProperty D` by a functor `C ⥤ D` -/
def inverseImage (P : MorphismProperty D) (F : C ⥤ D) : MorphismProperty C := fun _ _ f =>
  P (F.map f)


@[simp]
lemma inverseImage_iff (P : MorphismProperty D) (F : C ⥤ D) {X Y : C} (f : X ⟶ Y) :
                                           /-
                                             C : Type u
                                             inst✝¹ : CategoryTheory.Category.{v, u} C
                                             D : Type u_1
                                             inst✝ : CategoryTheory.Category.{u_2, u_1} D
                                             P : CategoryTheory.MorphismProperty D
                                             F : CategoryTheory.Functor C D
                                             X Y : C
                                             f : Quiver.Hom X Y
                                             ⊢ Iff (P.inverseImage F f) (P (F.map f))
                                           -/
    P.inverseImage F f ↔ P (F.map f) := by rfl
                                           /-
                                             🎉 no goals
                                           -/


/-- The image (up to isomorphisms) of a `MorphismProperty C` by a functor `C ⥤ D` -/
def map (P : MorphismProperty C) (F : C ⥤ D) : MorphismProperty D := fun _ _ f =>
  ∃ (X' Y' : C) (f' : X' ⟶ Y') (_ : P f'), Nonempty (Arrow.mk (F.map f') ≅ Arrow.mk f)


lemma map_mem_map (P : MorphismProperty C) (F : C ⥤ D) {X Y : C} (f : X ⟶ Y) (hf : P f) :
    (P.map F) (F.map f) := ⟨X, Y, f, hf, ⟨Iso.refl _⟩⟩


lemma monotone_map (F : C ⥤ D) :
    Monotone (map · F) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} D
    F : CategoryTheory.Functor C D
    ⊢ Monotone fun x => x.map F
  -/
  intro P Q h X Y f ⟨X', Y', f', hf', ⟨e⟩⟩
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} D
    F : CategoryTheory.Functor C D
    P Q : CategoryTheory.MorphismProperty C
    h : LE.le P Q
    X Y : D
    f : Quiver.Hom X Y
    X' Y' : C
    f' : Quiver.Hom X' Y'
    hf' : P f'
    e : CategoryTheory.Iso (CategoryTheory.Arrow.mk (F.map f')) (CategoryTheory.Ar …
    ⊢ Q.map F f
  -/
  exact ⟨X', Y', f', h _ hf', ⟨e⟩⟩
  /-
    🎉 no goals
  -/


lemma of_eq (P : MorphismProperty C) {X Y : C} {f : X ⟶ Y} (hf : P f)
    {X' Y' : C} {f' : X' ⟶ Y'}
    (hX : X = X') (hY : Y = Y') (h : f' = eqToHom hX.symm ≫ f ≫ eqToHom hY) :
    P f' := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.MorphismProperty C
    X Y : C
    f : Quiver.Hom X Y
    hf : P f
    X' Y' : C
    f' : Quiver.Hom X' Y'
    hX : Eq X X'
    hY : Eq Y Y'
    h : Eq f' (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (Cate …
    ⊢ P f'
  -/
  obtain rfl := hX
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.MorphismProperty C
    X Y : C
    f : Quiver.Hom X Y
    hf : P f
    Y' : C
    hY : Eq Y Y'
    f' : Quiver.Hom X Y'
    h : Eq f' (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (Cate …
    ⊢ P f'
  -/
  obtain rfl := hY
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.MorphismProperty C
    X Y : C
    f : Quiver.Hom X Y
    hf : P f
    f' : Quiver.Hom X Y
    h : Eq f' (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (Cate …
    ⊢ P f'
  -/
  obtain rfl : f' = f := by simpa using h
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.MorphismProperty C
    X Y : C
    f' : Quiver.Hom X Y
    hf : P f'
    h : Eq f' (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (Cate …
    ⊢ P f'
  -/
  exact hf
  /-
    🎉 no goals
  -/


/-- A morphism property `P` satisfies `P.RespectsRight Q` if it is stable under post-composition
with morphisms satisfying `Q`, i.e. whenever `P` holds for `f` and `Q` holds for `i` then `P`
holds for `f ≫ i`. -/
class RespectsRight (P Q : MorphismProperty C) : Prop where
  postcomp {X Y Z : C} (i : Y ⟶ Z) (hi : Q i) (f : X ⟶ Y) (hf : P f) : P (f ≫ i)


/-- A morphism property `P` satisfies `P.RespectsLeft Q` if it is stable under
pre-composition with morphisms satisfying `Q`, i.e. whenever `P` holds for `f`
and `Q` holds for `i` then `P` holds for `i ≫ f`. -/
class RespectsLeft (P Q : MorphismProperty C) : Prop where
  precomp {X Y Z : C} (i : X ⟶ Y) (hi : Q i) (f : Y ⟶ Z) (hf : P f) : P (i ≫ f)


/-- A morphism property `P` satisfies `P.Respects Q` if it is stable under composition on the
left and right by morphisms satisfying `Q`. -/
class Respects (P Q : MorphismProperty C) extends P.RespectsLeft Q, P.RespectsRight Q : Prop where


instance (P Q : MorphismProperty C) [P.RespectsLeft Q] [P.RespectsRight Q] : P.Respects Q where


instance (P Q : MorphismProperty C) [P.RespectsLeft Q] : P.op.RespectsRight Q.op where
  postcomp i hi f hf := RespectsLeft.precomp (Q := Q) i.unop hi f.unop hf


instance (P Q : MorphismProperty C) [P.RespectsRight Q] : P.op.RespectsLeft Q.op where
  precomp i hi f hf := RespectsRight.postcomp (Q := Q) i.unop hi f.unop hf


instance RespectsLeft.inf (P₁ P₂ Q : MorphismProperty C) [P₁.RespectsLeft Q]
    [P₂.RespectsLeft Q] : (P₁ ⊓ P₂).RespectsLeft Q where
  precomp i hi f hf := ⟨precomp i hi f hf.left, precomp i hi f hf.right⟩


instance RespectsRight.inf (P₁ P₂ Q : MorphismProperty C) [P₁.RespectsRight Q]
    [P₂.RespectsRight Q] : (P₁ ⊓ P₂).RespectsRight Q where
  postcomp i hi f hf := ⟨postcomp i hi f hf.left, postcomp i hi f hf.right⟩


/-- The `MorphismProperty C` satisfied by isomorphisms in `C`. -/
def isomorphisms : MorphismProperty C := fun _ _ f => IsIso f


/-- The `MorphismProperty C` satisfied by monomorphisms in `C`. -/
def monomorphisms : MorphismProperty C := fun _ _ f => Mono f


/-- The `MorphismProperty C` satisfied by epimorphisms in `C`. -/
def epimorphisms : MorphismProperty C := fun _ _ f => Epi f


/-- `P` respects isomorphisms, if it respects the morphism property `isomorphisms C`, i.e.
it is stable under pre- and postcomposition with isomorphisms. -/
abbrev RespectsIso (P : MorphismProperty C) : Prop := P.Respects (isomorphisms C)


lemma RespectsIso.mk (P : MorphismProperty C)
    (hprecomp : ∀ {X Y Z : C} (e : X ≅ Y) (f : Y ⟶ Z) (_ : P f), P (e.hom ≫ f))
    (hpostcomp : ∀ {X Y Z : C} (e : Y ≅ Z) (f : X ⟶ Y) (_ : P f), P (f ≫ e.hom)) :
    P.RespectsIso where
  precomp e (_ : IsIso e) f hf := hprecomp (asIso e) f hf
  postcomp e (_ : IsIso e) f hf := hpostcomp (asIso e) f hf


lemma RespectsIso.precomp (P : MorphismProperty C) [P.RespectsIso] {X Y Z : C} (e : X ⟶ Y)
    [IsIso e] (f : Y ⟶ Z) (hf : P f) : P (e ≫ f) :=
  RespectsLeft.precomp (Q := isomorphisms C) e ‹IsIso e› f hf


instance : RespectsIso (⊤ : MorphismProperty C) where
  precomp _ _ _ _ := trivial
  postcomp _ _ _ _ := trivial


lemma RespectsIso.postcomp (P : MorphismProperty C) [P.RespectsIso] {X Y Z : C} (e : Y ⟶ Z)
    [IsIso e] (f : X ⟶ Y) (hf : P f) : P (f ≫ e) :=
  RespectsRight.postcomp (Q := isomorphisms C) e ‹IsIso e› f hf


instance RespectsIso.op (P : MorphismProperty C) [RespectsIso P] : RespectsIso P.op where
  precomp e (_ : IsIso e) f hf := postcomp P e.unop f.unop hf
  postcomp e (_ : IsIso e) f hf := precomp P e.unop f.unop hf


instance RespectsIso.unop (P : MorphismProperty Cᵒᵖ) [RespectsIso P] : RespectsIso P.unop where
  precomp e (_ : IsIso e) f hf := postcomp P e.op f.op hf
  postcomp e (_ : IsIso e) f hf := precomp P e.op f.op hf


/-- The closure by isomorphisms of a `MorphismProperty` -/
def isoClosure (P : MorphismProperty C) : MorphismProperty C :=
  fun _ _ f => ∃ (Y₁ Y₂ : C) (f' : Y₁ ⟶ Y₂) (_ : P f'), Nonempty (Arrow.mk f' ≅ Arrow.mk f)


lemma le_isoClosure (P : MorphismProperty C) : P ≤ P.isoClosure :=
  fun _ _ f hf => ⟨_, _, f, hf, ⟨Iso.refl _⟩⟩


instance isoClosure_respectsIso (P : MorphismProperty C) :
    RespectsIso P.isoClosure where
  precomp := fun e (he : IsIso e) f ⟨_, _, f', hf', ⟨iso⟩⟩ => ⟨_, _, f', hf',
                                                                                   /-
                                                                                     C : Type u
                                                                                     inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                                     D : Type u_1
                                                                                     inst✝ : CategoryTheory.Category.{?u.11463, u_1} D
                                                                                     P : CategoryTheory.MorphismProperty C
                                                                                     X✝ Y✝ Z✝ : C
                                                                                     e : Quiver.Hom X✝ Y✝
                                                                                     he : CategoryTheory.IsIso e
                                                                                     f : Quiver.Hom Y✝ Z✝
                                                                                     x✝ : P.isoClosure f
                                                                                     w✝¹ w✝ : C
                                                                                     f' : Quiver.Hom w✝¹ w✝
                                                                                     hf' : P f'
                                                                                     iso : CategoryTheory.Iso (CategoryTheory.Arrow.mk f') (CategoryTheory.Arrow.mk …
                                                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.asIso iso.hom.left). …
                                                                                   -/
      ⟨Arrow.isoMk (asIso iso.hom.left ≪≫ asIso (inv e)) (asIso iso.hom.right) (by simp)⟩⟩
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
  postcomp := fun e (he : IsIso e) f ⟨_, _, f', hf', ⟨iso⟩⟩ => ⟨_, _, f', hf',
                                                                             /-
                                                                               C : Type u
                                                                               inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                               D : Type u_1
                                                                               inst✝ : CategoryTheory.Category.{?u.11463, u_1} D
                                                                               P : CategoryTheory.MorphismProperty C
                                                                               X✝ Y✝ Z✝ : C
                                                                               e : Quiver.Hom Y✝ Z✝
                                                                               he : CategoryTheory.IsIso e
                                                                               f : Quiver.Hom X✝ Y✝
                                                                               x✝ : P.isoClosure f
                                                                               w✝¹ w✝ : C
                                                                               f' : Quiver.Hom w✝¹ w✝
                                                                               hf' : P f'
                                                                               iso : CategoryTheory.Iso (CategoryTheory.Arrow.mk f') (CategoryTheory.Arrow.mk …
                                                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.asIso iso.hom.left).h …
                                                                             -/
      ⟨Arrow.isoMk (asIso iso.hom.left) (asIso iso.hom.right ≪≫ asIso e) (by simp)⟩⟩
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


lemma monotone_isoClosure : Monotone (isoClosure (C := C)) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ⊢ Monotone CategoryTheory.MorphismProperty.isoClosure
  -/
  intro P Q h X Y f ⟨X', Y', f', hf', ⟨e⟩⟩
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P Q : CategoryTheory.MorphismProperty C
    h : LE.le P Q
    X Y : C
    f : Quiver.Hom X Y
    X' Y' : C
    f' : Quiver.Hom X' Y'
    hf' : P f'
    e : CategoryTheory.Iso (CategoryTheory.Arrow.mk f') (CategoryTheory.Arrow.mk f)
    ⊢ Q.isoClosure f
  -/
  exact ⟨X', Y', f', h _ hf', ⟨e⟩⟩
  /-
    🎉 no goals
  -/


theorem cancel_left_of_respectsIso (P : MorphismProperty C) [hP : RespectsIso P] {X Y Z : C}
    (f : X ⟶ Y) (g : Y ⟶ Z) [IsIso f] : P (f ≫ g) ↔ P g :=
               /-
                 C : Type u
                 inst✝¹ : CategoryTheory.Category.{v, u} C
                 P : CategoryTheory.MorphismProperty C
                 hP : P.RespectsIso
                 X Y Z : C
                 f : Quiver.Hom X Y
                 g : Quiver.Hom Y Z
                 inst✝ : CategoryTheory.IsIso f
                 h : P (CategoryTheory.CategoryStruct.comp f g)
                 ⊢ P g
               -/
  ⟨fun h => by simpa using RespectsIso.precomp P (inv f) (f ≫ g) h, RespectsIso.precomp P f g⟩
               /-
                 🎉 no goals
               -/


theorem cancel_right_of_respectsIso (P : MorphismProperty C) [hP : RespectsIso P] {X Y Z : C}
    (f : X ⟶ Y) (g : Y ⟶ Z) [IsIso g] : P (f ≫ g) ↔ P f :=
               /-
                 C : Type u
                 inst✝¹ : CategoryTheory.Category.{v, u} C
                 P : CategoryTheory.MorphismProperty C
                 hP : P.RespectsIso
                 X Y Z : C
                 f : Quiver.Hom X Y
                 g : Quiver.Hom Y Z
                 inst✝ : CategoryTheory.IsIso g
                 h : P (CategoryTheory.CategoryStruct.comp f g)
                 ⊢ P f
               -/
  ⟨fun h => by simpa using RespectsIso.postcomp P (inv g) (f ≫ g) h, RespectsIso.postcomp P g f⟩
               /-
                 🎉 no goals
               -/


lemma comma_iso_iff (P : MorphismProperty C) [P.RespectsIso] {A B : Type*} [Category A] [Category B]
    {L : A ⥤ C} {R : B ⥤ C} {f g : Comma L R} (e : f ≅ g) :
    P f.hom ↔ P g.hom := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.MorphismProperty C
    inst✝² : P.RespectsIso
    A : Type u_2
    B : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} A
    inst✝ : CategoryTheory.Category.{u_5, u_3} B
    L : CategoryTheory.Functor A C
    R : CategoryTheory.Functor B C
    f g : CategoryTheory.Comma L R
    e : CategoryTheory.Iso f g
    ⊢ Iff (P f.hom) (P g.hom)
  -/
  simp [← Comma.inv_left_hom_right e.hom, cancel_left_of_respectsIso, cancel_right_of_respectsIso]
  /-
    🎉 no goals
  -/


theorem arrow_iso_iff (P : MorphismProperty C) [RespectsIso P] {f g : Arrow C}
    (e : f ≅ g) : P f.hom ↔ P g.hom :=
  P.comma_iso_iff e


theorem arrow_mk_iso_iff (P : MorphismProperty C) [RespectsIso P] {W X Y Z : C}
    {f : W ⟶ X} {g : Y ⟶ Z} (e : Arrow.mk f ≅ Arrow.mk g) : P f ↔ P g :=
  P.arrow_iso_iff e


theorem RespectsIso.of_respects_arrow_iso (P : MorphismProperty C)
    (hP : ∀ (f g : Arrow C) (_ : f ≅ g) (_ : P f.hom), P g.hom) : RespectsIso P where
  precomp {X Y Z} e (he : IsIso e) f hf := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.MorphismProperty C
      hP : ∀ (f g : CategoryTheory.Arrow C), CategoryTheory.Iso f g → P f.hom → P g. …
      X Y Z : C
      e : Quiver.Hom X Y
      he : CategoryTheory.IsIso e
      f : Quiver.Hom Y Z
      hf : P f
      ⊢ P (CategoryTheory.CategoryStruct.comp e f)
    -/
    refine hP (Arrow.mk f) (Arrow.mk (e ≫ f)) (Arrow.isoMk (asIso (inv e)) (Iso.refl _) ?_) hf
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.MorphismProperty C
      hP : ∀ (f g : CategoryTheory.Arrow C), CategoryTheory.Iso f g → P f.hom → P g. …
      X Y Z : C
      e : Quiver.Hom X Y
      he : CategoryTheory.IsIso e
      f : Quiver.Hom Y Z
      hf : P f
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.asIso (CategoryTheory …
    -/
    simp
    /-
      🎉 no goals
    -/
  postcomp {X Y Z} e (he : IsIso e) f hf := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.MorphismProperty C
      hP : ∀ (f g : CategoryTheory.Arrow C), CategoryTheory.Iso f g → P f.hom → P g. …
      X Y Z : C
      e : Quiver.Hom Y Z
      he : CategoryTheory.IsIso e
      f : Quiver.Hom X Y
      hf : P f
      ⊢ P (CategoryTheory.CategoryStruct.comp f e)
    -/
    refine hP (Arrow.mk f) (Arrow.mk (f ≫ e)) (Arrow.isoMk (Iso.refl _) (asIso e) ?_) hf
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.MorphismProperty C
      hP : ∀ (f g : CategoryTheory.Arrow C), CategoryTheory.Iso f g → P f.hom → P g. …
      X Y Z : C
      e : Quiver.Hom Y Z
      he : CategoryTheory.IsIso e
      f : Quiver.Hom X Y
      hf : P f
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl (CategoryThe …
    -/
    simp
    /-
      🎉 no goals
    -/


lemma isoClosure_eq_iff (P : MorphismProperty C) :
    P.isoClosure = P ↔ P.RespectsIso := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.MorphismProperty C
    ⊢ Iff (Eq P.isoClosure P) P.RespectsIso
  -/
  refine ⟨(· ▸ P.isoClosure_respectsIso), fun hP ↦ le_antisymm ?_ (P.le_isoClosure)⟩
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.MorphismProperty C
    hP : P.RespectsIso
    ⊢ LE.le P.isoClosure P
  -/
  intro X Y f ⟨X', Y', f', hf', ⟨e⟩⟩
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.MorphismProperty C
    hP : P.RespectsIso
    X Y : C
    f : Quiver.Hom X Y
    X' Y' : C
    f' : Quiver.Hom X' Y'
    hf' : P f'
    e : CategoryTheory.Iso (CategoryTheory.Arrow.mk f') (CategoryTheory.Arrow.mk f)
    ⊢ P f
  -/
  exact (P.arrow_mk_iso_iff e).1 hf'
  /-
    🎉 no goals
  -/


lemma isoClosure_eq_self (P : MorphismProperty C) [P.RespectsIso] :
                           /-
                             C : Type u
                             inst✝¹ : CategoryTheory.Category.{v, u} C
                             P : CategoryTheory.MorphismProperty C
                             inst✝ : P.RespectsIso
                             ⊢ Eq P.isoClosure P
                           -/
    P.isoClosure = P := by rwa [isoClosure_eq_iff]
                           /-
                             🎉 no goals
                           -/


@[simp]
lemma isoClosure_isoClosure (P : MorphismProperty C) :
    P.isoClosure.isoClosure = P.isoClosure :=
  P.isoClosure.isoClosure_eq_self


lemma isoClosure_le_iff (P Q : MorphismProperty C) [Q.RespectsIso] :
    P.isoClosure ≤ Q ↔ P ≤ Q := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P Q : CategoryTheory.MorphismProperty C
    inst✝ : Q.RespectsIso
    ⊢ Iff (LE.le P.isoClosure Q) (LE.le P Q)
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P Q : CategoryTheory.MorphismProperty C
      inst✝ : Q.RespectsIso
      ⊢ LE.le P.isoClosure Q → LE.le P Q
    -/
  · exact P.le_isoClosure.trans
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P Q : CategoryTheory.MorphismProperty C
      inst✝ : Q.RespectsIso
      ⊢ LE.le P Q → LE.le P.isoClosure Q
    -/
  · intro h
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      P Q : CategoryTheory.MorphismProperty C
      inst✝ : Q.RespectsIso
      h : LE.le P Q
      ⊢ LE.le P.isoClosure Q
    -/
    exact (monotone_isoClosure h).trans (by rw [Q.isoClosure_eq_self])
    /-
      🎉 no goals
    -/


instance map_respectsIso (P : MorphismProperty C) (F : C ⥤ D) :
    (P.map F).RespectsIso := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} D
    P : CategoryTheory.MorphismProperty C
    F : CategoryTheory.Functor C D
    ⊢ (P.map F).RespectsIso
  -/
  apply RespectsIso.of_respects_arrow_iso
  /-
    case hP
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} D
    P : CategoryTheory.MorphismProperty C
    F : CategoryTheory.Functor C D
    ⊢ ∀ (f g : CategoryTheory.Arrow D), CategoryTheory.Iso f g → P.map F f.hom → P …
  -/
  intro f g e ⟨X', Y', f', hf', ⟨e'⟩⟩
  /-
    case hP
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} D
    P : CategoryTheory.MorphismProperty C
    F : CategoryTheory.Functor C D
    f g : CategoryTheory.Arrow D
    e : CategoryTheory.Iso f g
    X' Y' : C
    f' : Quiver.Hom X' Y'
    hf' : P f'
    e' : CategoryTheory.Iso (CategoryTheory.Arrow.mk (F.map f')) (CategoryTheory.A …
    ⊢ P.map F g.hom
  -/
  exact ⟨X', Y', f', hf', ⟨e' ≪≫ e⟩⟩
  /-
    🎉 no goals
  -/


lemma map_le_iff (P : MorphismProperty C) {F : C ⥤ D} (Q : MorphismProperty D)
    [RespectsIso Q] :
    P.map F ≤ Q ↔ P ≤ Q.inverseImage F := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} D
    P : CategoryTheory.MorphismProperty C
    F : CategoryTheory.Functor C D
    Q : CategoryTheory.MorphismProperty D
    inst✝ : Q.RespectsIso
    ⊢ Iff (LE.le (P.map F) Q) (LE.le P (Q.inverseImage F))
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} D
      P : CategoryTheory.MorphismProperty C
      F : CategoryTheory.Functor C D
      Q : CategoryTheory.MorphismProperty D
      inst✝ : Q.RespectsIso
      ⊢ LE.le (P.map F) Q → LE.le P (Q.inverseImage F)
    -/
  · intro h X Y f hf
    /-
      case mp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} D
      P : CategoryTheory.MorphismProperty C
      F : CategoryTheory.Functor C D
      Q : CategoryTheory.MorphismProperty D
      inst✝ : Q.RespectsIso
      h : LE.le (P.map F) Q
      X Y : C
      f : Quiver.Hom X Y
      hf : P f
      ⊢ Q.inverseImage F f
    -/
    exact h (F.map f) (map_mem_map P F f hf)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} D
      P : CategoryTheory.MorphismProperty C
      F : CategoryTheory.Functor C D
      Q : CategoryTheory.MorphismProperty D
      inst✝ : Q.RespectsIso
      ⊢ LE.le P (Q.inverseImage F) → LE.le (P.map F) Q
    -/
  · intro h X Y f ⟨X', Y', f', hf', ⟨e⟩⟩
    /-
      case mpr
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} D
      P : CategoryTheory.MorphismProperty C
      F : CategoryTheory.Functor C D
      Q : CategoryTheory.MorphismProperty D
      inst✝ : Q.RespectsIso
      h : LE.le P (Q.inverseImage F)
      X Y : D
      f : Quiver.Hom X Y
      X' Y' : C
      f' : Quiver.Hom X' Y'
      hf' : P f'
      e : CategoryTheory.Iso (CategoryTheory.Arrow.mk (F.map f')) (CategoryTheory.Ar …
      ⊢ Q f
    -/
    exact (Q.arrow_mk_iso_iff e).1 (h _ hf')
    /-
      🎉 no goals
    -/


@[simp]
lemma map_isoClosure (P : MorphismProperty C) (F : C ⥤ D) :
    P.isoClosure.map F = P.map F := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} D
    P : CategoryTheory.MorphismProperty C
    F : CategoryTheory.Functor C D
    ⊢ Eq (P.isoClosure.map F) (P.map F)
  -/
  apply le_antisymm
    /-
      case a
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} D
      P : CategoryTheory.MorphismProperty C
      F : CategoryTheory.Functor C D
      ⊢ LE.le (P.isoClosure.map F) (P.map F)
    -/
  · rw [map_le_iff]
    /-
      case a
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} D
      P : CategoryTheory.MorphismProperty C
      F : CategoryTheory.Functor C D
      ⊢ LE.le P.isoClosure ((P.map F).inverseImage F)
    -/
    intro X Y f ⟨X', Y', f', hf', ⟨e⟩⟩
    /-
      case a
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} D
      P : CategoryTheory.MorphismProperty C
      F : CategoryTheory.Functor C D
      X Y : C
      f : Quiver.Hom X Y
      X' Y' : C
      f' : Quiver.Hom X' Y'
      hf' : P f'
      e : CategoryTheory.Iso (CategoryTheory.Arrow.mk f') (CategoryTheory.Arrow.mk f)
      ⊢ (P.map F).inverseImage F f
    -/
    exact ⟨_, _, f', hf', ⟨F.mapArrow.mapIso e⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case a
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} D
      P : CategoryTheory.MorphismProperty C
      F : CategoryTheory.Functor C D
      ⊢ LE.le (P.map F) (P.isoClosure.map F)
    -/
  · exact monotone_map _ (le_isoClosure P)
    /-
      🎉 no goals
    -/


lemma map_id_eq_isoClosure (P : MorphismProperty C) :
    P.map (𝟭 _) = P.isoClosure := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.MorphismProperty C
    ⊢ Eq (P.map (CategoryTheory.Functor.id C)) P.isoClosure
  -/
  apply le_antisymm
    /-
      case a
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.MorphismProperty C
      ⊢ LE.le (P.map (CategoryTheory.Functor.id C)) P.isoClosure
    -/
  · rw [map_le_iff]
    /-
      case a
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.MorphismProperty C
      ⊢ LE.le P (P.isoClosure.inverseImage (CategoryTheory.Functor.id C))
    -/
    intro X Y f hf
    /-
      case a
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.MorphismProperty C
      X Y : C
      f : Quiver.Hom X Y
      hf : P f
      ⊢ P.isoClosure.inverseImage (CategoryTheory.Functor.id C) f
    -/
    exact P.le_isoClosure _ hf
    /-
      🎉 no goals
    -/
    /-
      case a
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.MorphismProperty C
      ⊢ LE.le P.isoClosure (P.map (CategoryTheory.Functor.id C))
    -/
  · intro X Y f hf
    /-
      case a
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      P : CategoryTheory.MorphismProperty C
      X Y : C
      f : Quiver.Hom X Y
      hf : P.isoClosure f
      ⊢ P.map (CategoryTheory.Functor.id C) f
    -/
    exact hf
    /-
      🎉 no goals
    -/


lemma map_id (P : MorphismProperty C) [RespectsIso P] :
    P.map (𝟭 _) = P := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    P : CategoryTheory.MorphismProperty C
    inst✝ : P.RespectsIso
    ⊢ Eq (P.map (CategoryTheory.Functor.id C)) P
  -/
  rw [map_id_eq_isoClosure, P.isoClosure_eq_self]
  /-
    🎉 no goals
  -/


@[simp]
lemma map_map (P : MorphismProperty C) (F : C ⥤ D) {E : Type*} [Category E] (G : D ⥤ E) :
    (P.map F).map G = P.map (F ⋙ G) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} D
    P : CategoryTheory.MorphismProperty C
    F : CategoryTheory.Functor C D
    E : Type u_2
    inst✝ : CategoryTheory.Category.{u_4, u_2} E
    G : CategoryTheory.Functor D E
    ⊢ Eq ((P.map F).map G) (P.map (F.comp G))
  -/
  apply le_antisymm
    /-
      case a
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} D
      P : CategoryTheory.MorphismProperty C
      F : CategoryTheory.Functor C D
      E : Type u_2
      inst✝ : CategoryTheory.Category.{u_4, u_2} E
      G : CategoryTheory.Functor D E
      ⊢ LE.le ((P.map F).map G) (P.map (F.comp G))
    -/
  · rw [map_le_iff]
    /-
      case a
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} D
      P : CategoryTheory.MorphismProperty C
      F : CategoryTheory.Functor C D
      E : Type u_2
      inst✝ : CategoryTheory.Category.{u_4, u_2} E
      G : CategoryTheory.Functor D E
      ⊢ LE.le (P.map F) ((P.map (F.comp G)).inverseImage G)
    -/
    intro X Y f ⟨X', Y', f', hf', ⟨e⟩⟩
    /-
      case a
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} D
      P : CategoryTheory.MorphismProperty C
      F : CategoryTheory.Functor C D
      E : Type u_2
      inst✝ : CategoryTheory.Category.{u_4, u_2} E
      G : CategoryTheory.Functor D E
      X Y : D
      f : Quiver.Hom X Y
      X' Y' : C
      f' : Quiver.Hom X' Y'
      hf' : P f'
      e : CategoryTheory.Iso (CategoryTheory.Arrow.mk (F.map f')) (CategoryTheory.Ar …
      ⊢ (P.map (F.comp G)).inverseImage G f
    -/
    exact ⟨X', Y', f', hf', ⟨G.mapArrow.mapIso e⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case a
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} D
      P : CategoryTheory.MorphismProperty C
      F : CategoryTheory.Functor C D
      E : Type u_2
      inst✝ : CategoryTheory.Category.{u_4, u_2} E
      G : CategoryTheory.Functor D E
      ⊢ LE.le (P.map (F.comp G)) ((P.map F).map G)
    -/
  · rw [map_le_iff]
    /-
      case a
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} D
      P : CategoryTheory.MorphismProperty C
      F : CategoryTheory.Functor C D
      E : Type u_2
      inst✝ : CategoryTheory.Category.{u_4, u_2} E
      G : CategoryTheory.Functor D E
      ⊢ LE.le P (((P.map F).map G).inverseImage (F.comp G))
    -/
    intro X Y f hf
    /-
      case a
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} D
      P : CategoryTheory.MorphismProperty C
      F : CategoryTheory.Functor C D
      E : Type u_2
      inst✝ : CategoryTheory.Category.{u_4, u_2} E
      G : CategoryTheory.Functor D E
      X Y : C
      f : Quiver.Hom X Y
      hf : P f
      ⊢ ((P.map F).map G).inverseImage (F.comp G) f
    -/
    exact map_mem_map _ _ _ (map_mem_map _ _ _ hf)
    /-
      🎉 no goals
    -/


instance RespectsIso.inverseImage (P : MorphismProperty D) [RespectsIso P] (F : C ⥤ D) :
    RespectsIso (P.inverseImage F) where
  precomp {X Y Z} e (he : IsIso e) f hf := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} D
      P : CategoryTheory.MorphismProperty D
      inst✝ : P.RespectsIso
      F : CategoryTheory.Functor C D
      X Y Z : C
      e : Quiver.Hom X Y
      he : CategoryTheory.IsIso e
      f : Quiver.Hom Y Z
      hf : P.inverseImage F f
      ⊢ P.inverseImage F (CategoryTheory.CategoryStruct.comp e f)
    -/
    simpa [MorphismProperty.inverseImage, cancel_left_of_respectsIso] using hf
    /-
      🎉 no goals
    -/
  postcomp {X Y Z} e (he : IsIso e) f hf := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} D
      P : CategoryTheory.MorphismProperty D
      inst✝ : P.RespectsIso
      F : CategoryTheory.Functor C D
      X Y Z : C
      e : Quiver.Hom Y Z
      he : CategoryTheory.IsIso e
      f : Quiver.Hom X Y
      hf : P.inverseImage F f
      ⊢ P.inverseImage F (CategoryTheory.CategoryStruct.comp f e)
    -/
    simpa [MorphismProperty.inverseImage, cancel_right_of_respectsIso] using hf
    /-
      🎉 no goals
    -/


lemma map_eq_of_iso (P : MorphismProperty C) {F G : C ⥤ D} (e : F ≅ G) :
    P.map F = P.map G := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} D
    P : CategoryTheory.MorphismProperty C
    F G : CategoryTheory.Functor C D
    e : CategoryTheory.Iso F G
    ⊢ Eq (P.map F) (P.map G)
  -/
  revert F G e
  suffices ∀ {F G : C ⥤ D} (_ : F ≅ G), P.map F ≤ P.map G from
    fun F G e => le_antisymm (this e) (this e.symm)
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} D
    P : CategoryTheory.MorphismProperty C
    ⊢ ∀ {F G : CategoryTheory.Functor C D}, CategoryTheory.Iso F G → LE.le (P.map  …
  -/
  intro F G e X Y f ⟨X', Y', f', hf', ⟨e'⟩⟩
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} D
    P : CategoryTheory.MorphismProperty C
    F G : CategoryTheory.Functor C D
    e : CategoryTheory.Iso F G
    X Y : D
    f : Quiver.Hom X Y
    X' Y' : C
    f' : Quiver.Hom X' Y'
    hf' : P f'
    e' : CategoryTheory.Iso (CategoryTheory.Arrow.mk (F.map f')) (CategoryTheory.A …
    ⊢ P.map G f
  -/
  exact ⟨X', Y', f', hf', ⟨((Functor.mapArrowFunctor _ _).mapIso e.symm).app (Arrow.mk f') ≪≫ e'⟩⟩
  /-
    🎉 no goals
  -/


lemma map_inverseImage_le (P : MorphismProperty D) (F : C ⥤ D) :
    (P.inverseImage F).map F ≤ P.isoClosure :=
  fun _ _ _ ⟨_, _, f, hf, ⟨e⟩⟩ => ⟨_, _, F.map f, hf, ⟨e⟩⟩


lemma inverseImage_equivalence_inverse_eq_map_functor
    (P : MorphismProperty D) [RespectsIso P] (E : C ≌ D) :
    P.inverseImage E.functor = P.map E.inverse := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} D
    P : CategoryTheory.MorphismProperty D
    inst✝ : P.RespectsIso
    E : CategoryTheory.Equivalence C D
    ⊢ Eq (P.inverseImage E.functor) (P.map E.inverse)
  -/
  apply le_antisymm
    /-
      case a
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} D
      P : CategoryTheory.MorphismProperty D
      inst✝ : P.RespectsIso
      E : CategoryTheory.Equivalence C D
      ⊢ LE.le (P.inverseImage E.functor) (P.map E.inverse)
    -/
  · intro X Y f hf
    /-
      case a
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} D
      P : CategoryTheory.MorphismProperty D
      inst✝ : P.RespectsIso
      E : CategoryTheory.Equivalence C D
      X Y : C
      f : Quiver.Hom X Y
      hf : P.inverseImage E.functor f
      ⊢ P.map E.inverse f
    -/
    refine ⟨_, _, _, hf, ⟨?_⟩⟩
    /-
      case a
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} D
      P : CategoryTheory.MorphismProperty D
      inst✝ : P.RespectsIso
      E : CategoryTheory.Equivalence C D
      X Y : C
      f : Quiver.Hom X Y
      hf : P.inverseImage E.functor f
      ⊢ CategoryTheory.Iso (CategoryTheory.Arrow.mk (E.inverse.map (E.functor.map f) …
    -/
    exact ((Functor.mapArrowFunctor _ _).mapIso E.unitIso.symm).app (Arrow.mk f)
    /-
      🎉 no goals
    -/
    /-
      case a
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} D
      P : CategoryTheory.MorphismProperty D
      inst✝ : P.RespectsIso
      E : CategoryTheory.Equivalence C D
      ⊢ LE.le (P.map E.inverse) (P.inverseImage E.functor)
    -/
  · rw [map_le_iff]
    /-
      case a
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} D
      P : CategoryTheory.MorphismProperty D
      inst✝ : P.RespectsIso
      E : CategoryTheory.Equivalence C D
      ⊢ LE.le P ((P.inverseImage E.functor).inverseImage E.inverse)
    -/
    intro X Y f hf
    exact (P.arrow_mk_iso_iff
      (((Functor.mapArrowFunctor _ _).mapIso E.counitIso).app (Arrow.mk f))).2 hf


lemma inverseImage_equivalence_functor_eq_map_inverse
    (Q : MorphismProperty C) [RespectsIso Q] (E : C ≌ D) :
    Q.inverseImage E.inverse = Q.map E.functor :=
  inverseImage_equivalence_inverse_eq_map_functor Q E.symm


lemma map_inverseImage_eq_of_isEquivalence
    (P : MorphismProperty D) [P.RespectsIso] (F : C ⥤ D) [F.IsEquivalence] :
    (P.inverseImage F).map F = P := by
  erw [P.inverseImage_equivalence_inverse_eq_map_functor F.asEquivalence, map_map,
    P.map_eq_of_iso F.asEquivalence.counitIso, map_id]


lemma inverseImage_map_eq_of_isEquivalence
    (P : MorphismProperty C) [P.RespectsIso] (F : C ⥤ D) [F.IsEquivalence] :
    (P.map F).inverseImage F = P := by
  erw [((P.map F).inverseImage_equivalence_inverse_eq_map_functor (F.asEquivalence)), map_map,
    P.map_eq_of_iso F.asEquivalence.unitIso.symm, map_id]


@[simp]
                                                              /-
                                                                C : Type u
                                                                inst✝ : CategoryTheory.Category.{v, u} C
                                                                X Y : C
                                                                f : Quiver.Hom X Y
                                                                ⊢ Iff (CategoryTheory.MorphismProperty.isomorphisms C f) (CategoryTheory.IsIso …
                                                              -/
theorem isomorphisms.iff : (isomorphisms C) f ↔ IsIso f := by rfl
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
                                                               /-
                                                                 C : Type u
                                                                 inst✝ : CategoryTheory.Category.{v, u} C
                                                                 X Y : C
                                                                 f : Quiver.Hom X Y
                                                                 ⊢ Iff (CategoryTheory.MorphismProperty.monomorphisms C f) (CategoryTheory.Mono …
                                                               -/
theorem monomorphisms.iff : (monomorphisms C) f ↔ Mono f := by rfl
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp]
                                                            /-
                                                              C : Type u
                                                              inst✝ : CategoryTheory.Category.{v, u} C
                                                              X Y : C
                                                              f : Quiver.Hom X Y
                                                              ⊢ Iff (CategoryTheory.MorphismProperty.epimorphisms C f) (CategoryTheory.Epi f)
                                                            -/
theorem epimorphisms.iff : (epimorphisms C) f ↔ Epi f := by rfl
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem isomorphisms.infer_property [hf : IsIso f] : (isomorphisms C) f :=
  hf


theorem monomorphisms.infer_property [hf : Mono f] : (monomorphisms C) f :=
  hf


theorem epimorphisms.infer_property [hf : Epi f] : (epimorphisms C) f :=
  hf


instance RespectsIso.monomorphisms : RespectsIso (monomorphisms C) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝ : CategoryTheory.Category.{?u.42596, u_1} D
    ⊢ (CategoryTheory.MorphismProperty.monomorphisms C).RespectsIso
  -/
  apply RespectsIso.mk <;>
      /-
        case hprecomp
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        inst✝ : CategoryTheory.Category.{?u.42596, u_1} D
        ⊢ ∀ {X Y Z : C} (e : CategoryTheory.Iso X Y) (f : Quiver.Hom Y Z), CategoryThe …
      -/
      /-
        case hprecomp
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        inst✝ : CategoryTheory.Category.{?u.42596, u_1} D
        X Y Z : C
        e : CategoryTheory.Iso X Y
        f : Quiver.Hom Y Z
        ⊢ CategoryTheory.MorphismProperty.monomorphisms C f → CategoryTheory.MorphismP …
      -/
      /-
        case hprecomp
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        inst✝ : CategoryTheory.Category.{?u.42596, u_1} D
        X Y Z : C
        e : CategoryTheory.Iso X Y
        f : Quiver.Hom Y Z
        ⊢ CategoryTheory.Mono f → CategoryTheory.Mono (CategoryTheory.CategoryStruct.c …
      -/
      /-
        case hprecomp
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        inst✝ : CategoryTheory.Category.{?u.42596, u_1} D
        X Y Z : C
        e : CategoryTheory.Iso X Y
        f : Quiver.Hom Y Z
        x✝ : CategoryTheory.Mono f
        ⊢ CategoryTheory.Mono (CategoryTheory.CategoryStruct.comp e.hom f)
      -/
      /-
        🎉 no goals
      -/
      /-
        case hpostcomp
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        inst✝ : CategoryTheory.Category.{?u.42596, u_1} D
        X Y Z : C
        e : CategoryTheory.Iso Y Z
        f : Quiver.Hom X Y
        ⊢ CategoryTheory.Mono f → CategoryTheory.Mono (CategoryTheory.CategoryStruct.c …
      -/
      intro
      /-
        case hpostcomp
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        inst✝ : CategoryTheory.Category.{?u.42596, u_1} D
        X Y Z : C
        e : CategoryTheory.Iso Y Z
        f : Quiver.Hom X Y
        x✝ : CategoryTheory.Mono f
        ⊢ CategoryTheory.Mono (CategoryTheory.CategoryStruct.comp f e.hom)
      -/
      apply mono_comp
      /-
        🎉 no goals
      -/


instance RespectsIso.epimorphisms : RespectsIso (epimorphisms C) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝ : CategoryTheory.Category.{?u.43257, u_1} D
    ⊢ (CategoryTheory.MorphismProperty.epimorphisms C).RespectsIso
  -/
  apply RespectsIso.mk <;>
      /-
        case hprecomp
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        inst✝ : CategoryTheory.Category.{?u.43257, u_1} D
        ⊢ ∀ {X Y Z : C} (e : CategoryTheory.Iso X Y) (f : Quiver.Hom Y Z), CategoryThe …
      -/
      /-
        case hprecomp
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        inst✝ : CategoryTheory.Category.{?u.43257, u_1} D
        X Y Z : C
        e : CategoryTheory.Iso X Y
        f : Quiver.Hom Y Z
        ⊢ CategoryTheory.MorphismProperty.epimorphisms C f → CategoryTheory.MorphismPr …
      -/
      /-
        case hprecomp
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        inst✝ : CategoryTheory.Category.{?u.43257, u_1} D
        X Y Z : C
        e : CategoryTheory.Iso X Y
        f : Quiver.Hom Y Z
        ⊢ CategoryTheory.Epi f → CategoryTheory.Epi (CategoryTheory.CategoryStruct.com …
      -/
      /-
        case hprecomp
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        inst✝ : CategoryTheory.Category.{?u.43257, u_1} D
        X Y Z : C
        e : CategoryTheory.Iso X Y
        f : Quiver.Hom Y Z
        x✝ : CategoryTheory.Epi f
        ⊢ CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp e.hom f)
      -/
      /-
        🎉 no goals
      -/
      /-
        case hpostcomp
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        inst✝ : CategoryTheory.Category.{?u.43257, u_1} D
        X Y Z : C
        e : CategoryTheory.Iso Y Z
        f : Quiver.Hom X Y
        ⊢ CategoryTheory.Epi f → CategoryTheory.Epi (CategoryTheory.CategoryStruct.com …
      -/
      intro
      /-
        case hpostcomp
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        inst✝ : CategoryTheory.Category.{?u.43257, u_1} D
        X Y Z : C
        e : CategoryTheory.Iso Y Z
        f : Quiver.Hom X Y
        x✝ : CategoryTheory.Epi f
        ⊢ CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp f e.hom)
      -/
      apply epi_comp
      /-
        🎉 no goals
      -/


instance RespectsIso.isomorphisms : RespectsIso (isomorphisms C) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝ : CategoryTheory.Category.{?u.43918, u_1} D
    ⊢ (CategoryTheory.MorphismProperty.isomorphisms C).RespectsIso
  -/
  apply RespectsIso.mk <;>
      /-
        case hprecomp
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        inst✝ : CategoryTheory.Category.{?u.43918, u_1} D
        ⊢ ∀ {X Y Z : C} (e : CategoryTheory.Iso X Y) (f : Quiver.Hom Y Z), CategoryThe …
      -/
      /-
        case hprecomp
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        inst✝ : CategoryTheory.Category.{?u.43918, u_1} D
        X Y Z : C
        e : CategoryTheory.Iso X Y
        f : Quiver.Hom Y Z
        ⊢ CategoryTheory.MorphismProperty.isomorphisms C f → CategoryTheory.MorphismPr …
      -/
      /-
        case hprecomp
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        inst✝ : CategoryTheory.Category.{?u.43918, u_1} D
        X Y Z : C
        e : CategoryTheory.Iso X Y
        f : Quiver.Hom Y Z
        ⊢ CategoryTheory.IsIso f → CategoryTheory.IsIso (CategoryTheory.CategoryStruct …
      -/
      /-
        case hprecomp
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        inst✝ : CategoryTheory.Category.{?u.43918, u_1} D
        X Y Z : C
        e : CategoryTheory.Iso X Y
        f : Quiver.Hom Y Z
        x✝ : CategoryTheory.IsIso f
        ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp e.hom f)
      -/
      /-
        🎉 no goals
      -/
      /-
        case hpostcomp
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        inst✝ : CategoryTheory.Category.{?u.43918, u_1} D
        X Y Z : C
        e : CategoryTheory.Iso Y Z
        f : Quiver.Hom X Y
        ⊢ CategoryTheory.IsIso f → CategoryTheory.IsIso (CategoryTheory.CategoryStruct …
      -/
      intro
      /-
        case hpostcomp
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        inst✝ : CategoryTheory.Category.{?u.43918, u_1} D
        X Y Z : C
        e : CategoryTheory.Iso Y Z
        f : Quiver.Hom X Y
        x✝ : CategoryTheory.IsIso f
        ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp f e.hom)
      -/
      exact IsIso.comp_isIso
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-07-02")] alias RespectsIso.cancel_left_isIso :=
  cancel_left_of_respectsIso

@[deprecated (since := "2024-07-02")] alias RespectsIso.cancel_right_isIso :=
  cancel_right_of_respectsIso

@[deprecated (since := "2024-07-02")] alias RespectsIso.arrow_iso_iff := arrow_iso_iff

@[deprecated (since := "2024-07-02")] alias RespectsIso.arrow_mk_iso_iff := arrow_mk_iso_iff

@[deprecated (since := "2024-07-02")] alias RespectsIso.isoClosure_eq := isoClosure_eq_self


/-- If `W₁` and `W₂` are morphism properties on two categories `C₁` and `C₂`,
this is the induced morphism property on `C₁ × C₂`. -/
def prod {C₁ C₂ : Type*} [Category C₁] [Category C₂]
    (W₁ : MorphismProperty C₁) (W₂ : MorphismProperty C₂) :
    MorphismProperty (C₁ × C₂) :=
  fun _ _ f => W₁ f.1 ∧ W₂ f.2


/-- If `W j` are morphism properties on categories `C j` for all `j`, this is the
induced morphism property on the category `∀ j, C j`. -/
def pi {J : Type w} {C : J → Type u} [∀ j, Category.{v} (C j)]
    (W : ∀ j, MorphismProperty (C j)) : MorphismProperty (∀ j, C j) :=
  fun _ _ f => ∀ j, (W j) (f j)


/-- The morphism property on `J ⥤ C` which is defined objectwise
from `W : MorphismProperty C`. -/
def functorCategory (W : MorphismProperty C) (J : Type*) [Category J] :
    MorphismProperty (J ⥤ C) :=
  fun _ _ f => ∀ (j : J), W (f.app j)


/-- Given `W : MorphismProperty C`, this is the morphism property on `Arrow C` of morphisms
whose left and right parts are in `W`. -/
def arrow (W : MorphismProperty C) :
    MorphismProperty (Arrow C) :=
  fun _ _ f => W f.left ∧ W f.right


lemma isIso_app_iff_of_iso {F G : C ⥤ D} (α : F ⟶ G) {X Y : C} (e : X ≅ Y) :
    IsIso (α.app X) ↔ IsIso (α.app Y) :=
  (MorphismProperty.isomorphisms D).arrow_mk_iso_iff
                                               /-
                                                 C : Type u
                                                 inst✝¹ : CategoryTheory.Category.{v, u} C
                                                 D : Type u_1
                                                 inst✝ : CategoryTheory.Category.{u_2, u_1} D
                                                 F G : CategoryTheory.Functor C D
                                                 α : Quiver.Hom F G
                                                 X Y : C
                                                 e : CategoryTheory.Iso X Y
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.mapIso e).hom (CategoryTheory.Arro …
                                               -/
    (Arrow.isoMk (F.mapIso e) (G.mapIso e) (by simp))
                                               /-
                                                 🎉 no goals
                                               -/


