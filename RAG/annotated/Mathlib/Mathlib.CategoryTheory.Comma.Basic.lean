/-- The objects of the comma category are triples of an object `left : A`, an object
   `right : B` and a morphism `hom : L.obj left ⟶ R.obj right`. -/
structure Comma (L : A ⥤ T) (R : B ⥤ T) : Type max u₁ u₂ v₃ where
  left : A
  right : B
  hom : L.obj left ⟶ R.obj right

-- Satisfying the inhabited linter

instance Comma.inhabited [Inhabited T] : Inhabited (Comma (𝟭 T) (𝟭 T)) where
  default :=
    { left := default
      right := default
      hom := 𝟙 default }


/-- A morphism between two objects in the comma category is a commutative square connecting the
    morphisms coming from the two objects using morphisms in the image of the functors `L` and `R`.
-/
@[ext]
structure CommaMorphism (X Y : Comma L R) where
  left : X.left ⟶ Y.left
  right : X.right ⟶ Y.right
  w : L.map left ≫ Y.hom = X.hom ≫ R.map right := by aesop_cat

-- Satisfying the inhabited linter

instance CommaMorphism.inhabited [Inhabited (Comma L R)] :
    Inhabited (CommaMorphism (default : Comma L R) default) :=
    ⟨{ left := 𝟙 _, right := 𝟙 _}⟩


attribute [reassoc (attr := simp)] CommaMorphism.w


instance commaCategory : Category (Comma L R) where
  Hom X Y := CommaMorphism X Y
  id X :=
    { left := 𝟙 X.left
      right := 𝟙 X.right }
  comp f g :=
    { left := f.left ≫ g.left
      right := f.right ≫ g.right }


@[ext]
lemma hom_ext (f g : X ⟶ Y) (h₁ : f.left = g.left) (h₂ : f.right = g.right) : f = g :=
  CommaMorphism.ext h₁ h₂


@[simp]
theorem id_left : (𝟙 X : CommaMorphism X X).left = 𝟙 X.left :=
  rfl


@[simp]
theorem id_right : (𝟙 X : CommaMorphism X X).right = 𝟙 X.right :=
  rfl


@[simp]
theorem comp_left : (f ≫ g).left = f.left ≫ g.left :=
  rfl


@[simp]
theorem comp_right : (f ≫ g).right = f.right ≫ g.right :=
  rfl


/-- The functor sending an object `X` in the comma category to `X.left`. -/
@[simps]
def fst : Comma L R ⥤ A where
  obj X := X.left
  map f := f.left


/-- The functor sending an object `X` in the comma category to `X.right`. -/
@[simps]
def snd : Comma L R ⥤ B where
  obj X := X.right
  map f := f.right


/-- We can interpret the commutative square constituting a morphism in the comma category as a
    natural transformation between the functors `fst ⋙ L` and `snd ⋙ R` from the comma category
    to `T`, where the components are given by the morphism that constitutes an object of the comma
    category. -/
@[simps]
def natTrans : fst L R ⋙ L ⟶ snd L R ⋙ R where app X := X.hom


@[simp]
theorem eqToHom_left (X Y : Comma L R) (H : X = Y) :
                                                 /-
                                                   A : Type u₁
                                                   inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
                                                   B : Type u₂
                                                   inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
                                                   T : Type u₃
                                                   inst✝³ : CategoryTheory.Category.{v₃, u₃} T
                                                   A' : Type u₄
                                                   inst✝² : CategoryTheory.Category.{v₄, u₄} A'
                                                   B' : Type u₅
                                                   inst✝¹ : CategoryTheory.Category.{v₅, u₅} B'
                                                   T' : Type u₆
                                                   inst✝ : CategoryTheory.Category.{v₆, u₆} T'
                                                   L : CategoryTheory.Functor A T
                                                   R : CategoryTheory.Functor B T
                                                   X Y : CategoryTheory.Comma L R
                                                   H : Eq X Y
                                                   ⊢ Eq X.left Y.left
                                                 -/
    CommaMorphism.left (eqToHom H) = eqToHom (by cases H; rfl) := by
                                                          /-
                                                            🎉 no goals
                                                          -/
  /-
    A : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    X Y : CategoryTheory.Comma L R
    H : Eq X Y
    ⊢ Eq (CategoryTheory.eqToHom H).left (CategoryTheory.eqToHom ⋯)
  -/
  cases H
  /-
    case refl
    A : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    X : CategoryTheory.Comma L R
    ⊢ Eq (CategoryTheory.eqToHom ⋯).left (CategoryTheory.eqToHom ⋯)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem eqToHom_right (X Y : Comma L R) (H : X = Y) :
                                                  /-
                                                    A : Type u₁
                                                    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
                                                    B : Type u₂
                                                    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
                                                    T : Type u₃
                                                    inst✝³ : CategoryTheory.Category.{v₃, u₃} T
                                                    A' : Type u₄
                                                    inst✝² : CategoryTheory.Category.{v₄, u₄} A'
                                                    B' : Type u₅
                                                    inst✝¹ : CategoryTheory.Category.{v₅, u₅} B'
                                                    T' : Type u₆
                                                    inst✝ : CategoryTheory.Category.{v₆, u₆} T'
                                                    L : CategoryTheory.Functor A T
                                                    R : CategoryTheory.Functor B T
                                                    X Y : CategoryTheory.Comma L R
                                                    H : Eq X Y
                                                    ⊢ Eq X.right Y.right
                                                  -/
    CommaMorphism.right (eqToHom H) = eqToHom (by cases H; rfl) := by
                                                           /-
                                                             🎉 no goals
                                                           -/
  /-
    A : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    X Y : CategoryTheory.Comma L R
    H : Eq X Y
    ⊢ Eq (CategoryTheory.eqToHom H).right (CategoryTheory.eqToHom ⋯)
  -/
  cases H
  /-
    case refl
    A : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    X : CategoryTheory.Comma L R
    ⊢ Eq (CategoryTheory.eqToHom ⋯).right (CategoryTheory.eqToHom ⋯)
  -/
  rfl
  /-
    🎉 no goals
  -/


instance [IsIso e] : IsIso e.left :=
  (Comma.fst L R).map_isIso e


instance [IsIso e] : IsIso e.right :=
  (Comma.snd L R).map_isIso e


@[simp]
lemma inv_left [IsIso e] : (inv e).left = inv e.left := by
  /-
    A : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    X Y : CategoryTheory.Comma L R
    e : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso e
    ⊢ Eq (CategoryTheory.inv e).left (CategoryTheory.inv e.left)
  -/
  apply IsIso.eq_inv_of_hom_inv_id
  /-
    case hom_inv_id
    A : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    X Y : CategoryTheory.Comma L R
    e : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso e
    ⊢ Eq (CategoryTheory.CategoryStruct.comp e.left (CategoryTheory.inv e).left) ( …
  -/
  rw [← Comma.comp_left, IsIso.hom_inv_id, id_left]
  /-
    🎉 no goals
  -/


@[simp]
lemma inv_right [IsIso e] : (inv e).right = inv e.right := by
  /-
    A : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    X Y : CategoryTheory.Comma L R
    e : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso e
    ⊢ Eq (CategoryTheory.inv e).right (CategoryTheory.inv e.right)
  -/
  apply IsIso.eq_inv_of_hom_inv_id
  /-
    case hom_inv_id
    A : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    X Y : CategoryTheory.Comma L R
    e : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso e
    ⊢ Eq (CategoryTheory.CategoryStruct.comp e.right (CategoryTheory.inv e).right) …
  -/
  rw [← Comma.comp_right, IsIso.hom_inv_id, id_right]
  /-
    🎉 no goals
  -/


lemma left_hom_inv_right [IsIso e] : L.map (e.left) ≫ Y.hom ≫ R.map (inv e.right) = X.hom := by
  /-
    A : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    X Y : CategoryTheory.Comma L R
    e : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso e
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map e.left) (CategoryTheory.Catego …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma inv_left_hom_right [IsIso e] : L.map (inv e.left) ≫ X.hom ≫ R.map e.right = Y.hom := by
  /-
    A : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    X Y : CategoryTheory.Comma L R
    e : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso e
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map (CategoryTheory.inv e.left)) ( …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Extract the isomorphism between the left objects from an isomorphism in the comma category. -/
@[simps!]
def leftIso {X Y : Comma L₁ R₁} (α : X ≅ Y) : X.left ≅ Y.left := (fst L₁ R₁).mapIso α


/-- Extract the isomorphism between the right objects from an isomorphism in the comma category. -/
@[simps!]
def rightIso {X Y : Comma L₁ R₁} (α : X ≅ Y) : X.right ≅ Y.right := (snd L₁ R₁).mapIso α


/-- Construct an isomorphism in the comma category given isomorphisms of the objects whose forward
directions give a commutative square.
-/
@[simps]
def isoMk {X Y : Comma L₁ R₁} (l : X.left ≅ Y.left) (r : X.right ≅ Y.right)
    (h : L₁.map l.hom ≫ Y.hom = X.hom ≫ R₁.map r.hom := by aesop_cat) : X ≅ Y where
  hom :=
    { left := l.hom
      right := r.hom
      w := h }
  inv :=
    { left := l.inv
      right := r.inv
      w := by
        rw [← L₁.mapIso_inv l, Iso.inv_comp_eq, L₁.mapIso_hom, ← Category.assoc, h,
          Category.assoc, ← R₁.map_comp]
        /-
          A : Type u₁
          inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
          B : Type u₂
          inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
          T : Type u₃
          inst✝³ : CategoryTheory.Category.{v₃, u₃} T
          A' : Type u₄
          inst✝² : CategoryTheory.Category.{v₄, u₄} A'
          B' : Type u₅
          inst✝¹ : CategoryTheory.Category.{v₅, u₅} B'
          T' : Type u₆
          inst✝ : CategoryTheory.Category.{v₆, u₆} T'
          L : CategoryTheory.Functor A T
          R : CategoryTheory.Functor B T
          L₁ L₂ L₃ : CategoryTheory.Functor A T
          R₁ R₂ R₃ : CategoryTheory.Functor B T
          X Y : CategoryTheory.Comma L₁ R₁
          l : CategoryTheory.Iso X.left Y.left
          r : CategoryTheory.Iso X.right Y.right
          h : autoParam (Eq (CategoryTheory.CategoryStruct.comp (L₁.map l.hom) Y.hom) (C …
          ⊢ Eq X.hom (CategoryTheory.CategoryStruct.comp X.hom (R₁.map (CategoryTheory.C …
        -/
        simp }
        /-
          🎉 no goals
        -/


/-- The functor `Comma L R ⥤ Comma L' R'` induced by three functors `F₁`, `F₂`, `F`
and two natural transformations `F₁ ⋙ L' ⟶ L ⋙ F` and `R ⋙ F ⟶ F₂ ⋙ R'`. -/
@[simps]
def map : Comma L R ⥤ Comma L' R' where
  obj X :=
    { left := F₁.obj X.left
      right := F₂.obj X.right
      hom := α.app X.left ≫ F.map X.hom ≫ β.app X.right }
  map {X Y} φ :=
    { left := F₁.map φ.left
      right := F₂.map φ.right
      w := by
        /-
          A : Type u₁
          inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
          B : Type u₂
          inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
          T : Type u₃
          inst✝³ : CategoryTheory.Category.{v₃, u₃} T
          A' : Type u₄
          inst✝² : CategoryTheory.Category.{v₄, u₄} A'
          B' : Type u₅
          inst✝¹ : CategoryTheory.Category.{v₅, u₅} B'
          T' : Type u₆
          inst✝ : CategoryTheory.Category.{v₆, u₆} T'
          L : CategoryTheory.Functor A T
          R : CategoryTheory.Functor B T
          L₁ L₂ L₃ : CategoryTheory.Functor A T
          R₁ R₂ R₃ : CategoryTheory.Functor B T
          L' : CategoryTheory.Functor A' T'
          R' : CategoryTheory.Functor B' T'
          F₁ : CategoryTheory.Functor A A'
          F₂ : CategoryTheory.Functor B B'
          F : CategoryTheory.Functor T T'
          α : Quiver.Hom (F₁.comp L') (L.comp F)
          β : Quiver.Hom (R.comp F) (F₂.comp R')
          X Y : CategoryTheory.Comma L R
          φ : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (L'.map (F₁.map φ.left)) ((fun X => { …
        -/
        dsimp
        /-
          A : Type u₁
          inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
          B : Type u₂
          inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
          T : Type u₃
          inst✝³ : CategoryTheory.Category.{v₃, u₃} T
          A' : Type u₄
          inst✝² : CategoryTheory.Category.{v₄, u₄} A'
          B' : Type u₅
          inst✝¹ : CategoryTheory.Category.{v₅, u₅} B'
          T' : Type u₆
          inst✝ : CategoryTheory.Category.{v₆, u₆} T'
          L : CategoryTheory.Functor A T
          R : CategoryTheory.Functor B T
          L₁ L₂ L₃ : CategoryTheory.Functor A T
          R₁ R₂ R₃ : CategoryTheory.Functor B T
          L' : CategoryTheory.Functor A' T'
          R' : CategoryTheory.Functor B' T'
          F₁ : CategoryTheory.Functor A A'
          F₂ : CategoryTheory.Functor B B'
          F : CategoryTheory.Functor T T'
          α : Quiver.Hom (F₁.comp L') (L.comp F)
          β : Quiver.Hom (R.comp F) (F₂.comp R')
          X Y : CategoryTheory.Comma L R
          φ : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (L'.map (F₁.map φ.left)) (CategoryThe …
        -/
        rw [assoc, assoc]
        /-
          A : Type u₁
          inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
          B : Type u₂
          inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
          T : Type u₃
          inst✝³ : CategoryTheory.Category.{v₃, u₃} T
          A' : Type u₄
          inst✝² : CategoryTheory.Category.{v₄, u₄} A'
          B' : Type u₅
          inst✝¹ : CategoryTheory.Category.{v₅, u₅} B'
          T' : Type u₆
          inst✝ : CategoryTheory.Category.{v₆, u₆} T'
          L : CategoryTheory.Functor A T
          R : CategoryTheory.Functor B T
          L₁ L₂ L₃ : CategoryTheory.Functor A T
          R₁ R₂ R₃ : CategoryTheory.Functor B T
          L' : CategoryTheory.Functor A' T'
          R' : CategoryTheory.Functor B' T'
          F₁ : CategoryTheory.Functor A A'
          F₂ : CategoryTheory.Functor B B'
          F : CategoryTheory.Functor T T'
          α : Quiver.Hom (F₁.comp L') (L.comp F)
          β : Quiver.Hom (R.comp F) (F₂.comp R')
          X Y : CategoryTheory.Comma L R
          φ : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (L'.map (F₁.map φ.left)) (CategoryThe …
        -/
        erw [α.naturality_assoc, ← β.naturality]
        /-
          A : Type u₁
          inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
          B : Type u₂
          inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
          T : Type u₃
          inst✝³ : CategoryTheory.Category.{v₃, u₃} T
          A' : Type u₄
          inst✝² : CategoryTheory.Category.{v₄, u₄} A'
          B' : Type u₅
          inst✝¹ : CategoryTheory.Category.{v₅, u₅} B'
          T' : Type u₆
          inst✝ : CategoryTheory.Category.{v₆, u₆} T'
          L : CategoryTheory.Functor A T
          R : CategoryTheory.Functor B T
          L₁ L₂ L₃ : CategoryTheory.Functor A T
          R₁ R₂ R₃ : CategoryTheory.Functor B T
          L' : CategoryTheory.Functor A' T'
          R' : CategoryTheory.Functor B' T'
          F₁ : CategoryTheory.Functor A A'
          F₂ : CategoryTheory.Functor B B'
          F : CategoryTheory.Functor T T'
          α : Quiver.Hom (F₁.comp L') (L.comp F)
          β : Quiver.Hom (R.comp F) (F₂.comp R')
          X Y : CategoryTheory.Comma L R
          φ : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.app X.left) (CategoryTheory.Catego …
        -/
        dsimp
        /-
          A : Type u₁
          inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
          B : Type u₂
          inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
          T : Type u₃
          inst✝³ : CategoryTheory.Category.{v₃, u₃} T
          A' : Type u₄
          inst✝² : CategoryTheory.Category.{v₄, u₄} A'
          B' : Type u₅
          inst✝¹ : CategoryTheory.Category.{v₅, u₅} B'
          T' : Type u₆
          inst✝ : CategoryTheory.Category.{v₆, u₆} T'
          L : CategoryTheory.Functor A T
          R : CategoryTheory.Functor B T
          L₁ L₂ L₃ : CategoryTheory.Functor A T
          R₁ R₂ R₃ : CategoryTheory.Functor B T
          L' : CategoryTheory.Functor A' T'
          R' : CategoryTheory.Functor B' T'
          F₁ : CategoryTheory.Functor A A'
          F₂ : CategoryTheory.Functor B B'
          F : CategoryTheory.Functor T T'
          α : Quiver.Hom (F₁.comp L') (L.comp F)
          β : Quiver.Hom (R.comp F) (F₂.comp R')
          X Y : CategoryTheory.Comma L R
          φ : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.app X.left) (CategoryTheory.Catego …
        -/
        rw [← F.map_comp_assoc, ← F.map_comp_assoc, φ.w] }
        /-
          🎉 no goals
        -/


instance faithful_map [F₁.Faithful] [F₂.Faithful] : (map α β).Faithful where
  map_injective {X Y} f g h := by
    /-
      A : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} A
      B : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} B
      T : Type u₃
      inst✝⁵ : CategoryTheory.Category.{v₃, u₃} T
      A' : Type u₄
      inst✝⁴ : CategoryTheory.Category.{v₄, u₄} A'
      B' : Type u₅
      inst✝³ : CategoryTheory.Category.{v₅, u₅} B'
      T' : Type u₆
      inst✝² : CategoryTheory.Category.{v₆, u₆} T'
      L : CategoryTheory.Functor A T
      R : CategoryTheory.Functor B T
      L₁ L₂ L₃ : CategoryTheory.Functor A T
      R₁ R₂ R₃ : CategoryTheory.Functor B T
      L' : CategoryTheory.Functor A' T'
      R' : CategoryTheory.Functor B' T'
      F₁ : CategoryTheory.Functor A A'
      F₂ : CategoryTheory.Functor B B'
      F : CategoryTheory.Functor T T'
      α : Quiver.Hom (F₁.comp L') (L.comp F)
      β : Quiver.Hom (R.comp F) (F₂.comp R')
      inst✝¹ : F₁.Faithful
      inst✝ : F₂.Faithful
      X Y : CategoryTheory.Comma L R
      f g : Quiver.Hom X Y
      h : Eq ((CategoryTheory.Comma.map α β).map f) ((CategoryTheory.Comma.map α β). …
      ⊢ Eq f g
    -/
    ext
      /-
        case h₁
        A : Type u₁
        inst✝⁷ : CategoryTheory.Category.{v₁, u₁} A
        B : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} B
        T : Type u₃
        inst✝⁵ : CategoryTheory.Category.{v₃, u₃} T
        A' : Type u₄
        inst✝⁴ : CategoryTheory.Category.{v₄, u₄} A'
        B' : Type u₅
        inst✝³ : CategoryTheory.Category.{v₅, u₅} B'
        T' : Type u₆
        inst✝² : CategoryTheory.Category.{v₆, u₆} T'
        L : CategoryTheory.Functor A T
        R : CategoryTheory.Functor B T
        L₁ L₂ L₃ : CategoryTheory.Functor A T
        R₁ R₂ R₃ : CategoryTheory.Functor B T
        L' : CategoryTheory.Functor A' T'
        R' : CategoryTheory.Functor B' T'
        F₁ : CategoryTheory.Functor A A'
        F₂ : CategoryTheory.Functor B B'
        F : CategoryTheory.Functor T T'
        α : Quiver.Hom (F₁.comp L') (L.comp F)
        β : Quiver.Hom (R.comp F) (F₂.comp R')
        inst✝¹ : F₁.Faithful
        inst✝ : F₂.Faithful
        X Y : CategoryTheory.Comma L R
        f g : Quiver.Hom X Y
        h : Eq ((CategoryTheory.Comma.map α β).map f) ((CategoryTheory.Comma.map α β). …
        ⊢ Eq f.left g.left
      -/
    · exact F₁.map_injective (congr_arg CommaMorphism.left h)
      /-
        🎉 no goals
      -/
      /-
        case h₂
        A : Type u₁
        inst✝⁷ : CategoryTheory.Category.{v₁, u₁} A
        B : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} B
        T : Type u₃
        inst✝⁵ : CategoryTheory.Category.{v₃, u₃} T
        A' : Type u₄
        inst✝⁴ : CategoryTheory.Category.{v₄, u₄} A'
        B' : Type u₅
        inst✝³ : CategoryTheory.Category.{v₅, u₅} B'
        T' : Type u₆
        inst✝² : CategoryTheory.Category.{v₆, u₆} T'
        L : CategoryTheory.Functor A T
        R : CategoryTheory.Functor B T
        L₁ L₂ L₃ : CategoryTheory.Functor A T
        R₁ R₂ R₃ : CategoryTheory.Functor B T
        L' : CategoryTheory.Functor A' T'
        R' : CategoryTheory.Functor B' T'
        F₁ : CategoryTheory.Functor A A'
        F₂ : CategoryTheory.Functor B B'
        F : CategoryTheory.Functor T T'
        α : Quiver.Hom (F₁.comp L') (L.comp F)
        β : Quiver.Hom (R.comp F) (F₂.comp R')
        inst✝¹ : F₁.Faithful
        inst✝ : F₂.Faithful
        X Y : CategoryTheory.Comma L R
        f g : Quiver.Hom X Y
        h : Eq ((CategoryTheory.Comma.map α β).map f) ((CategoryTheory.Comma.map α β). …
        ⊢ Eq f.right g.right
      -/
    · exact F₂.map_injective (congr_arg CommaMorphism.right h)
      /-
        🎉 no goals
      -/


instance full_map [F.Faithful] [F₁.Full] [F₂.Full] [IsIso α] [IsIso β] : (map α β).Full where
  map_surjective {X Y} φ :=
   ⟨{ left := F₁.preimage φ.left
      right := F₂.preimage φ.right
      w := F.map_injective (by
        rw [← cancel_mono (β.app _), ← cancel_epi (α.app _), F.map_comp, F.map_comp,
          assoc, assoc]
        /-
          A : Type u₁
          inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} A
          B : Type u₂
          inst✝⁹ : CategoryTheory.Category.{v₂, u₂} B
          T : Type u₃
          inst✝⁸ : CategoryTheory.Category.{v₃, u₃} T
          A' : Type u₄
          inst✝⁷ : CategoryTheory.Category.{v₄, u₄} A'
          B' : Type u₅
          inst✝⁶ : CategoryTheory.Category.{v₅, u₅} B'
          T' : Type u₆
          inst✝⁵ : CategoryTheory.Category.{v₆, u₆} T'
          L : CategoryTheory.Functor A T
          R : CategoryTheory.Functor B T
          L₁ L₂ L₃ : CategoryTheory.Functor A T
          R₁ R₂ R₃ : CategoryTheory.Functor B T
          L' : CategoryTheory.Functor A' T'
          R' : CategoryTheory.Functor B' T'
          F₁ : CategoryTheory.Functor A A'
          F₂ : CategoryTheory.Functor B B'
          F : CategoryTheory.Functor T T'
          α : Quiver.Hom (F₁.comp L') (L.comp F)
          β : Quiver.Hom (R.comp F) (F₂.comp R')
          inst✝⁴ : F.Faithful
          inst✝³ : F₁.Full
          inst✝² : F₂.Full
          inst✝¹ : CategoryTheory.IsIso α
          inst✝ : CategoryTheory.IsIso β
          X Y : CategoryTheory.Comma L R
          φ : Quiver.Hom ((CategoryTheory.Comma.map α β).obj X) ((CategoryTheory.Comma.m …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.app X.left) (CategoryTheory.Catego …
        -/
        erw [← α.naturality_assoc, β.naturality]
        /-
          A : Type u₁
          inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} A
          B : Type u₂
          inst✝⁹ : CategoryTheory.Category.{v₂, u₂} B
          T : Type u₃
          inst✝⁸ : CategoryTheory.Category.{v₃, u₃} T
          A' : Type u₄
          inst✝⁷ : CategoryTheory.Category.{v₄, u₄} A'
          B' : Type u₅
          inst✝⁶ : CategoryTheory.Category.{v₅, u₅} B'
          T' : Type u₆
          inst✝⁵ : CategoryTheory.Category.{v₆, u₆} T'
          L : CategoryTheory.Functor A T
          R : CategoryTheory.Functor B T
          L₁ L₂ L₃ : CategoryTheory.Functor A T
          R₁ R₂ R₃ : CategoryTheory.Functor B T
          L' : CategoryTheory.Functor A' T'
          R' : CategoryTheory.Functor B' T'
          F₁ : CategoryTheory.Functor A A'
          F₂ : CategoryTheory.Functor B B'
          F : CategoryTheory.Functor T T'
          α : Quiver.Hom (F₁.comp L') (L.comp F)
          β : Quiver.Hom (R.comp F) (F₂.comp R')
          inst✝⁴ : F.Faithful
          inst✝³ : F₁.Full
          inst✝² : F₂.Full
          inst✝¹ : CategoryTheory.IsIso α
          inst✝ : CategoryTheory.IsIso β
          X Y : CategoryTheory.Comma L R
          φ : Quiver.Hom ((CategoryTheory.Comma.map α β).obj X) ((CategoryTheory.Comma.m …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F₁.comp L').map (F₁.preimage φ.left …
        -/
        dsimp
        /-
          A : Type u₁
          inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} A
          B : Type u₂
          inst✝⁹ : CategoryTheory.Category.{v₂, u₂} B
          T : Type u₃
          inst✝⁸ : CategoryTheory.Category.{v₃, u₃} T
          A' : Type u₄
          inst✝⁷ : CategoryTheory.Category.{v₄, u₄} A'
          B' : Type u₅
          inst✝⁶ : CategoryTheory.Category.{v₅, u₅} B'
          T' : Type u₆
          inst✝⁵ : CategoryTheory.Category.{v₆, u₆} T'
          L : CategoryTheory.Functor A T
          R : CategoryTheory.Functor B T
          L₁ L₂ L₃ : CategoryTheory.Functor A T
          R₁ R₂ R₃ : CategoryTheory.Functor B T
          L' : CategoryTheory.Functor A' T'
          R' : CategoryTheory.Functor B' T'
          F₁ : CategoryTheory.Functor A A'
          F₂ : CategoryTheory.Functor B B'
          F : CategoryTheory.Functor T T'
          α : Quiver.Hom (F₁.comp L') (L.comp F)
          β : Quiver.Hom (R.comp F) (F₂.comp R')
          inst✝⁴ : F.Faithful
          inst✝³ : F₁.Full
          inst✝² : F₂.Full
          inst✝¹ : CategoryTheory.IsIso α
          inst✝ : CategoryTheory.IsIso β
          X Y : CategoryTheory.Comma L R
          φ : Quiver.Hom ((CategoryTheory.Comma.map α β).obj X) ((CategoryTheory.Comma.m …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (L'.map (F₁.map (F₁.preimage φ.left)) …
        -/
        rw [F₁.map_preimage, F₂.map_preimage]
        /-
          A : Type u₁
          inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} A
          B : Type u₂
          inst✝⁹ : CategoryTheory.Category.{v₂, u₂} B
          T : Type u₃
          inst✝⁸ : CategoryTheory.Category.{v₃, u₃} T
          A' : Type u₄
          inst✝⁷ : CategoryTheory.Category.{v₄, u₄} A'
          B' : Type u₅
          inst✝⁶ : CategoryTheory.Category.{v₅, u₅} B'
          T' : Type u₆
          inst✝⁵ : CategoryTheory.Category.{v₆, u₆} T'
          L : CategoryTheory.Functor A T
          R : CategoryTheory.Functor B T
          L₁ L₂ L₃ : CategoryTheory.Functor A T
          R₁ R₂ R₃ : CategoryTheory.Functor B T
          L' : CategoryTheory.Functor A' T'
          R' : CategoryTheory.Functor B' T'
          F₁ : CategoryTheory.Functor A A'
          F₂ : CategoryTheory.Functor B B'
          F : CategoryTheory.Functor T T'
          α : Quiver.Hom (F₁.comp L') (L.comp F)
          β : Quiver.Hom (R.comp F) (F₂.comp R')
          inst✝⁴ : F.Faithful
          inst✝³ : F₁.Full
          inst✝² : F₂.Full
          inst✝¹ : CategoryTheory.IsIso α
          inst✝ : CategoryTheory.IsIso β
          X Y : CategoryTheory.Comma L R
          φ : Quiver.Hom ((CategoryTheory.Comma.map α β).obj X) ((CategoryTheory.Comma.m …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (L'.map φ.left) (CategoryTheory.Categ …
        -/
        /-
          🎉 no goals
        -/
        simpa using φ.w) }, by aesop_cat⟩
                               /-
                                 🎉 no goals
                               -/


instance essSurj_map [F₁.EssSurj] [F₂.EssSurj] [F.Full] [IsIso α] [IsIso β] :
    (map α β).EssSurj where
  mem_essImage X :=
    ⟨{  left := F₁.objPreimage X.left
        right := F₂.objPreimage X.right
        hom := F.preimage ((inv α).app _ ≫ L'.map (F₁.objObjPreimageIso X.left).hom ≫
          X.hom ≫ R'.map (F₂.objObjPreimageIso X.right).inv ≫ (inv β).app _) },
            ⟨isoMk (F₁.objObjPreimageIso X.left) (F₂.objObjPreimageIso X.right) (by
              /-
                A : Type u₁
                inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} A
                B : Type u₂
                inst✝⁹ : CategoryTheory.Category.{v₂, u₂} B
                T : Type u₃
                inst✝⁸ : CategoryTheory.Category.{v₃, u₃} T
                A' : Type u₄
                inst✝⁷ : CategoryTheory.Category.{v₄, u₄} A'
                B' : Type u₅
                inst✝⁶ : CategoryTheory.Category.{v₅, u₅} B'
                T' : Type u₆
                inst✝⁵ : CategoryTheory.Category.{v₆, u₆} T'
                L : CategoryTheory.Functor A T
                R : CategoryTheory.Functor B T
                L₁ L₂ L₃ : CategoryTheory.Functor A T
                R₁ R₂ R₃ : CategoryTheory.Functor B T
                L' : CategoryTheory.Functor A' T'
                R' : CategoryTheory.Functor B' T'
                F₁ : CategoryTheory.Functor A A'
                F₂ : CategoryTheory.Functor B B'
                F : CategoryTheory.Functor T T'
                α : Quiver.Hom (F₁.comp L') (L.comp F)
                β : Quiver.Hom (R.comp F) (F₂.comp R')
                inst✝⁴ : F₁.EssSurj
                inst✝³ : F₂.EssSurj
                inst✝² : F.Full
                inst✝¹ : CategoryTheory.IsIso α
                inst✝ : CategoryTheory.IsIso β
                X : CategoryTheory.Comma L' R'
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (L'.map (F₁.objObjPreimageIso X.left) …
              -/
              dsimp
              simp only [NatIso.isIso_inv_app, Functor.comp_obj, Functor.map_preimage, assoc,
                IsIso.inv_hom_id, comp_id, IsIso.hom_inv_id_assoc]
              /-
                A : Type u₁
                inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} A
                B : Type u₂
                inst✝⁹ : CategoryTheory.Category.{v₂, u₂} B
                T : Type u₃
                inst✝⁸ : CategoryTheory.Category.{v₃, u₃} T
                A' : Type u₄
                inst✝⁷ : CategoryTheory.Category.{v₄, u₄} A'
                B' : Type u₅
                inst✝⁶ : CategoryTheory.Category.{v₅, u₅} B'
                T' : Type u₆
                inst✝⁵ : CategoryTheory.Category.{v₆, u₆} T'
                L : CategoryTheory.Functor A T
                R : CategoryTheory.Functor B T
                L₁ L₂ L₃ : CategoryTheory.Functor A T
                R₁ R₂ R₃ : CategoryTheory.Functor B T
                L' : CategoryTheory.Functor A' T'
                R' : CategoryTheory.Functor B' T'
                F₁ : CategoryTheory.Functor A A'
                F₂ : CategoryTheory.Functor B B'
                F : CategoryTheory.Functor T T'
                α : Quiver.Hom (F₁.comp L') (L.comp F)
                β : Quiver.Hom (R.comp F) (F₂.comp R')
                inst✝⁴ : F₁.EssSurj
                inst✝³ : F₂.EssSurj
                inst✝² : F.Full
                inst✝¹ : CategoryTheory.IsIso α
                inst✝ : CategoryTheory.IsIso β
                X : CategoryTheory.Comma L' R'
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (L'.map (F₁.objObjPreimageIso X.left) …
              -/
              rw [← R'.map_comp, Iso.inv_hom_id, R'.map_id, comp_id])⟩⟩
              /-
                🎉 no goals
              -/


noncomputable instance isEquivalenceMap
    [F₁.IsEquivalence] [F₂.IsEquivalence] [F.Faithful] [F.Full] [IsIso α] [IsIso β] :
    (map α β).IsEquivalence where


/-- The equality between `map α β ⋙ fst L' R'` and `fst L R ⋙ F₁`,
where `α : F₁ ⋙ L' ⟶ L ⋙ F`. -/
@[simp]
theorem map_fst : map α β ⋙ fst L' R' = fst L R ⋙ F₁ :=
  rfl


/-- The isomorphism between `map α β ⋙ fst L' R'` and `fst L R ⋙ F₁`,
where `α : F₁ ⋙ L' ⟶ L ⋙ F`. -/
@[simps!]
def mapFst : map α β ⋙ fst L' R' ≅ fst L R ⋙ F₁ :=
                                                /-
                                                  A : Type u₁
                                                  inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
                                                  B : Type u₂
                                                  inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
                                                  T : Type u₃
                                                  inst✝³ : CategoryTheory.Category.{v₃, u₃} T
                                                  A' : Type u₄
                                                  inst✝² : CategoryTheory.Category.{v₄, u₄} A'
                                                  B' : Type u₅
                                                  inst✝¹ : CategoryTheory.Category.{v₅, u₅} B'
                                                  T' : Type u₆
                                                  inst✝ : CategoryTheory.Category.{v₆, u₆} T'
                                                  L : CategoryTheory.Functor A T
                                                  R : CategoryTheory.Functor B T
                                                  L₁ L₂ L₃ : CategoryTheory.Functor A T
                                                  R₁ R₂ R₃ : CategoryTheory.Functor B T
                                                  L' : CategoryTheory.Functor A' T'
                                                  R' : CategoryTheory.Functor B' T'
                                                  F₁ : CategoryTheory.Functor A A'
                                                  F₂ : CategoryTheory.Functor B B'
                                                  F : CategoryTheory.Functor T T'
                                                  α : Quiver.Hom (F₁.comp L') (L.comp F)
                                                  β : Quiver.Hom (R.comp F) (F₂.comp R')
                                                  ⊢ ∀ {X Y : CategoryTheory.Comma L R} (f : Quiver.Hom X Y), Eq (CategoryTheory. …
                                                -/
  NatIso.ofComponents (fun _ => Iso.refl _) (by aesop_cat)
                                                /-
                                                  🎉 no goals
                                                -/


/-- The equality between `map α β ⋙ snd L' R'` and `snd L R ⋙ F₂`,
where `β : R ⋙ F ⟶ F₂ ⋙ R'`. -/
@[simp]
theorem map_snd : map α β ⋙ snd L' R' = snd L R ⋙ F₂ :=
  rfl


/-- The isomorphism between `map α β ⋙ snd L' R'` and `snd L R ⋙ F₂`,
where `β : R ⋙ F ⟶ F₂ ⋙ R'`. -/
@[simps!]
def mapSnd : map α β ⋙ snd L' R' ≅ snd L R ⋙ F₂ :=
                                                /-
                                                  A : Type u₁
                                                  inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
                                                  B : Type u₂
                                                  inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
                                                  T : Type u₃
                                                  inst✝³ : CategoryTheory.Category.{v₃, u₃} T
                                                  A' : Type u₄
                                                  inst✝² : CategoryTheory.Category.{v₄, u₄} A'
                                                  B' : Type u₅
                                                  inst✝¹ : CategoryTheory.Category.{v₅, u₅} B'
                                                  T' : Type u₆
                                                  inst✝ : CategoryTheory.Category.{v₆, u₆} T'
                                                  L : CategoryTheory.Functor A T
                                                  R : CategoryTheory.Functor B T
                                                  L₁ L₂ L₃ : CategoryTheory.Functor A T
                                                  R₁ R₂ R₃ : CategoryTheory.Functor B T
                                                  L' : CategoryTheory.Functor A' T'
                                                  R' : CategoryTheory.Functor B' T'
                                                  F₁ : CategoryTheory.Functor A A'
                                                  F₂ : CategoryTheory.Functor B B'
                                                  F : CategoryTheory.Functor T T'
                                                  α : Quiver.Hom (F₁.comp L') (L.comp F)
                                                  β : Quiver.Hom (R.comp F) (F₂.comp R')
                                                  ⊢ ∀ {X Y : CategoryTheory.Comma L R} (f : Quiver.Hom X Y), Eq (CategoryTheory. …
                                                -/
  NatIso.ofComponents (fun _ => Iso.refl _) (by aesop_cat)
                                                /-
                                                  🎉 no goals
                                                -/


/-- A natural transformation `L₁ ⟶ L₂` induces a functor `Comma L₂ R ⥤ Comma L₁ R`. -/
@[simps]
def mapLeft (l : L₁ ⟶ L₂) : Comma L₂ R ⥤ Comma L₁ R where
  obj X :=
    { left := X.left
      right := X.right
      hom := l.app X.left ≫ X.hom }
  map f :=
    { left := f.left
      right := f.right }


/-- The functor `Comma L R ⥤ Comma L R` induced by the identity natural transformation on `L` is
    naturally isomorphic to the identity functor. -/
@[simps!]
def mapLeftId : mapLeft R (𝟙 L) ≅ 𝟭 _ :=
                                /-
                                  A : Type u₁
                                  inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
                                  B : Type u₂
                                  inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
                                  T : Type u₃
                                  inst✝³ : CategoryTheory.Category.{v₃, u₃} T
                                  A' : Type u₄
                                  inst✝² : CategoryTheory.Category.{v₄, u₄} A'
                                  B' : Type u₅
                                  inst✝¹ : CategoryTheory.Category.{v₅, u₅} B'
                                  T' : Type u₆
                                  inst✝ : CategoryTheory.Category.{v₆, u₆} T'
                                  L : CategoryTheory.Functor A T
                                  R : CategoryTheory.Functor B T
                                  L₁ L₂ L₃ : CategoryTheory.Functor A T
                                  R₁ R₂ R₃ : CategoryTheory.Functor B T
                                  X : CategoryTheory.Comma L R
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map (CategoryTheory.Iso.refl ((Cat …
                                -/
                                /-
                                  🎉 no goals
                                -/
  NatIso.ofComponents (fun X => isoMk (Iso.refl _) (Iso.refl _))
  /-
    🎉 no goals
  -/


/-- The functor `Comma L₁ R ⥤ Comma L₃ R` induced by the composition of two natural transformations
    `l : L₁ ⟶ L₂` and `l' : L₂ ⟶ L₃` is naturally isomorphic to the composition of the two functors
    induced by these natural transformations. -/
@[simps!]
def mapLeftComp (l : L₁ ⟶ L₂) (l' : L₂ ⟶ L₃) :
    mapLeft R (l ≫ l') ≅ mapLeft R l' ⋙ mapLeft R l :=
                                /-
                                  A : Type u₁
                                  inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
                                  B : Type u₂
                                  inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
                                  T : Type u₃
                                  inst✝³ : CategoryTheory.Category.{v₃, u₃} T
                                  A' : Type u₄
                                  inst✝² : CategoryTheory.Category.{v₄, u₄} A'
                                  B' : Type u₅
                                  inst✝¹ : CategoryTheory.Category.{v₅, u₅} B'
                                  T' : Type u₆
                                  inst✝ : CategoryTheory.Category.{v₆, u₆} T'
                                  L : CategoryTheory.Functor A T
                                  R : CategoryTheory.Functor B T
                                  L₁ L₂ L₃ : CategoryTheory.Functor A T
                                  R₁ R₂ R₃ : CategoryTheory.Functor B T
                                  l : Quiver.Hom L₁ L₂
                                  l' : Quiver.Hom L₂ L₃
                                  X : CategoryTheory.Comma L₃ R
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (L₁.map (CategoryTheory.Iso.refl ((Ca …
                                -/
                                /-
                                  🎉 no goals
                                -/
  NatIso.ofComponents (fun X => isoMk (Iso.refl _) (Iso.refl _))
  /-
    🎉 no goals
  -/


/-- Two equal natural transformations `L₁ ⟶ L₂` yield naturally isomorphic functors
    `Comma L₁ R ⥤ Comma L₂ R`. -/
@[simps!]
def mapLeftEq (l l' : L₁ ⟶ L₂) (h : l = l') : mapLeft R l ≅ mapLeft R l' :=
                                /-
                                  A : Type u₁
                                  inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
                                  B : Type u₂
                                  inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
                                  T : Type u₃
                                  inst✝³ : CategoryTheory.Category.{v₃, u₃} T
                                  A' : Type u₄
                                  inst✝² : CategoryTheory.Category.{v₄, u₄} A'
                                  B' : Type u₅
                                  inst✝¹ : CategoryTheory.Category.{v₅, u₅} B'
                                  T' : Type u₆
                                  inst✝ : CategoryTheory.Category.{v₆, u₆} T'
                                  L : CategoryTheory.Functor A T
                                  R : CategoryTheory.Functor B T
                                  L₁ L₂ L₃ : CategoryTheory.Functor A T
                                  R₁ R₂ R₃ : CategoryTheory.Functor B T
                                  l l' : Quiver.Hom L₁ L₂
                                  h : Eq l l'
                                  X : CategoryTheory.Comma L₂ R
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (L₁.map (CategoryTheory.Iso.refl ((Ca …
                                -/
                                /-
                                  🎉 no goals
                                -/
  NatIso.ofComponents (fun X => isoMk (Iso.refl _) (Iso.refl _))
  /-
    🎉 no goals
  -/


/-- A natural isomorphism `L₁ ≅ L₂` induces an equivalence of categories
    `Comma L₁ R ≌ Comma L₂ R`. -/
@[simps!]
def mapLeftIso (i : L₁ ≅ L₂) : Comma L₁ R ≌ Comma L₂ R where
  functor := mapLeft _ i.inv
  inverse := mapLeft _ i.hom
  unitIso := (mapLeftId _ _).symm ≪≫ mapLeftEq _ _ _ i.hom_inv_id.symm ≪≫ mapLeftComp _ _ _
  counitIso := (mapLeftComp _ _ _).symm ≪≫ mapLeftEq _ _ _ i.inv_hom_id ≪≫ mapLeftId _ _


/-- A natural transformation `R₁ ⟶ R₂` induces a functor `Comma L R₁ ⥤ Comma L R₂`. -/
@[simps]
def mapRight (r : R₁ ⟶ R₂) : Comma L R₁ ⥤ Comma L R₂ where
  obj X :=
    { left := X.left
      right := X.right
      hom := X.hom ≫ r.app X.right }
  map f :=
    { left := f.left
      right := f.right }


/-- The functor `Comma L R ⥤ Comma L R` induced by the identity natural transformation on `R` is
    naturally isomorphic to the identity functor. -/
@[simps!]
def mapRightId : mapRight L (𝟙 R) ≅ 𝟭 _ :=
                                /-
                                  A : Type u₁
                                  inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
                                  B : Type u₂
                                  inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
                                  T : Type u₃
                                  inst✝³ : CategoryTheory.Category.{v₃, u₃} T
                                  A' : Type u₄
                                  inst✝² : CategoryTheory.Category.{v₄, u₄} A'
                                  B' : Type u₅
                                  inst✝¹ : CategoryTheory.Category.{v₅, u₅} B'
                                  T' : Type u₆
                                  inst✝ : CategoryTheory.Category.{v₆, u₆} T'
                                  L : CategoryTheory.Functor A T
                                  R : CategoryTheory.Functor B T
                                  L₁ L₂ L₃ : CategoryTheory.Functor A T
                                  R₁ R₂ R₃ : CategoryTheory.Functor B T
                                  X : CategoryTheory.Comma L R
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map (CategoryTheory.Iso.refl ((Cat …
                                -/
                                /-
                                  🎉 no goals
                                -/
  NatIso.ofComponents (fun X => isoMk (Iso.refl _) (Iso.refl _))
  /-
    🎉 no goals
  -/


/-- The functor `Comma L R₁ ⥤ Comma L R₃` induced by the composition of the natural transformations
    `r : R₁ ⟶ R₂` and `r' : R₂ ⟶ R₃` is naturally isomorphic to the composition of the functors
    induced by these natural transformations. -/
@[simps!]
def mapRightComp (r : R₁ ⟶ R₂) (r' : R₂ ⟶ R₃) :
    mapRight L (r ≫ r') ≅ mapRight L r ⋙ mapRight L r' :=
                                /-
                                  A : Type u₁
                                  inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
                                  B : Type u₂
                                  inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
                                  T : Type u₃
                                  inst✝³ : CategoryTheory.Category.{v₃, u₃} T
                                  A' : Type u₄
                                  inst✝² : CategoryTheory.Category.{v₄, u₄} A'
                                  B' : Type u₅
                                  inst✝¹ : CategoryTheory.Category.{v₅, u₅} B'
                                  T' : Type u₆
                                  inst✝ : CategoryTheory.Category.{v₆, u₆} T'
                                  L : CategoryTheory.Functor A T
                                  R : CategoryTheory.Functor B T
                                  L₁ L₂ L₃ : CategoryTheory.Functor A T
                                  R₁ R₂ R₃ : CategoryTheory.Functor B T
                                  r : Quiver.Hom R₁ R₂
                                  r' : Quiver.Hom R₂ R₃
                                  X : CategoryTheory.Comma L R₁
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map (CategoryTheory.Iso.refl ((Cat …
                                -/
                                /-
                                  🎉 no goals
                                -/
  NatIso.ofComponents (fun X => isoMk (Iso.refl _) (Iso.refl _))
  /-
    🎉 no goals
  -/


/-- Two equal natural transformations `R₁ ⟶ R₂` yield naturally isomorphic functors
    `Comma L R₁ ⥤ Comma L R₂`. -/
@[simps!]
def mapRightEq (r r' : R₁ ⟶ R₂) (h : r = r') : mapRight L r ≅ mapRight L r' :=
                                /-
                                  A : Type u₁
                                  inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
                                  B : Type u₂
                                  inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
                                  T : Type u₃
                                  inst✝³ : CategoryTheory.Category.{v₃, u₃} T
                                  A' : Type u₄
                                  inst✝² : CategoryTheory.Category.{v₄, u₄} A'
                                  B' : Type u₅
                                  inst✝¹ : CategoryTheory.Category.{v₅, u₅} B'
                                  T' : Type u₆
                                  inst✝ : CategoryTheory.Category.{v₆, u₆} T'
                                  L : CategoryTheory.Functor A T
                                  R : CategoryTheory.Functor B T
                                  L₁ L₂ L₃ : CategoryTheory.Functor A T
                                  R₁ R₂ R₃ : CategoryTheory.Functor B T
                                  r r' : Quiver.Hom R₁ R₂
                                  h : Eq r r'
                                  X : CategoryTheory.Comma L R₁
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map (CategoryTheory.Iso.refl ((Cat …
                                -/
                                /-
                                  🎉 no goals
                                -/
  NatIso.ofComponents (fun X => isoMk (Iso.refl _) (Iso.refl _))
  /-
    🎉 no goals
  -/


/-- A natural isomorphism `R₁ ≅ R₂` induces an equivalence of categories
    `Comma L R₁ ≌ Comma L R₂`. -/
@[simps!]
def mapRightIso (i : R₁ ≅ R₂) : Comma L R₁ ≌ Comma L R₂ where
  functor := mapRight _ i.hom
  inverse := mapRight _ i.inv
  unitIso := (mapRightId _ _).symm ≪≫ mapRightEq _ _ _ i.hom_inv_id.symm ≪≫ mapRightComp _ _ _
  counitIso := (mapRightComp _ _ _).symm ≪≫ mapRightEq _ _ _ i.inv_hom_id ≪≫ mapRightId _ _


/-- The functor `(F ⋙ L, R) ⥤ (L, R)` -/
@[simps]
def preLeft (F : C ⥤ A) (L : A ⥤ T) (R : B ⥤ T) : Comma (F ⋙ L) R ⥤ Comma L R where
  obj X :=
    { left := F.obj X.left
      right := X.right
      hom := X.hom }
  map f :=
    { left := F.map f.left
      right := f.right
              /-
                A : Type u₁
                inst✝⁶ : CategoryTheory.Category.{v₁, u₁} A
                B : Type u₂
                inst✝⁵ : CategoryTheory.Category.{v₂, u₂} B
                T : Type u₃
                inst✝⁴ : CategoryTheory.Category.{v₃, u₃} T
                A' : Type u₄
                inst✝³ : CategoryTheory.Category.{v₄, u₄} A'
                B' : Type u₅
                inst✝² : CategoryTheory.Category.{v₅, u₅} B'
                T' : Type u₆
                inst✝¹ : CategoryTheory.Category.{v₆, u₆} T'
                L✝ : CategoryTheory.Functor A T
                R✝ : CategoryTheory.Functor B T
                C : Type u₄
                inst✝ : CategoryTheory.Category.{v₄, u₄} C
                F : CategoryTheory.Functor C A
                L : CategoryTheory.Functor A T
                R : CategoryTheory.Functor B T
                X✝ Y✝ : CategoryTheory.Comma (F.comp L) R
                f : Quiver.Hom X✝ Y✝
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map (F.map f.left)) ((fun X => { l …
              -/
      w := by simpa using f.w }
              /-
                🎉 no goals
              -/


/-- `Comma.preLeft` is a particular case of `Comma.map`,
but with better definitional properties. -/
def preLeftIso (F : C ⥤ A) (L : A ⥤ T) (R : B ⥤ T) :
    preLeft F L R ≅ map (F ⋙ L).rightUnitor.inv (R.rightUnitor.hom ≫ R.leftUnitor.inv) :=
                                /-
                                  A : Type u₁
                                  inst✝⁶ : CategoryTheory.Category.{v₁, u₁} A
                                  B : Type u₂
                                  inst✝⁵ : CategoryTheory.Category.{v₂, u₂} B
                                  T : Type u₃
                                  inst✝⁴ : CategoryTheory.Category.{v₃, u₃} T
                                  A' : Type u₄
                                  inst✝³ : CategoryTheory.Category.{v₄, u₄} A'
                                  B' : Type u₅
                                  inst✝² : CategoryTheory.Category.{v₅, u₅} B'
                                  T' : Type u₆
                                  inst✝¹ : CategoryTheory.Category.{v₆, u₆} T'
                                  L✝ : CategoryTheory.Functor A T
                                  R✝ : CategoryTheory.Functor B T
                                  C : Type u₄
                                  inst✝ : CategoryTheory.Category.{v₄, u₄} C
                                  F : CategoryTheory.Functor C A
                                  L : CategoryTheory.Functor A T
                                  R : CategoryTheory.Functor B T
                                  X : CategoryTheory.Comma (F.comp L) R
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map (CategoryTheory.Iso.refl ((Cat …
                                -/
                                /-
                                  🎉 no goals
                                -/
  NatIso.ofComponents (fun X => isoMk (Iso.refl _) (Iso.refl _))
  /-
    🎉 no goals
  -/


instance (F : C ⥤ A) (L : A ⥤ T) (R : B ⥤ T) [F.Faithful] : (preLeft F L R).Faithful :=
  Functor.Faithful.of_iso (preLeftIso F L R).symm


instance (F : C ⥤ A) (L : A ⥤ T) (R : B ⥤ T) [F.Full] : (preLeft F L R).Full :=
  Functor.Full.of_iso (preLeftIso F L R).symm


instance (F : C ⥤ A) (L : A ⥤ T) (R : B ⥤ T) [F.EssSurj] : (preLeft F L R).EssSurj :=
  Functor.essSurj_of_iso (preLeftIso F L R).symm


/-- If `F` is an equivalence, then so is `preLeft F L R`. -/
instance isEquivalence_preLeft (F : C ⥤ A) (L : A ⥤ T) (R : B ⥤ T) [F.IsEquivalence] :
    (preLeft F L R).IsEquivalence where


/-- The functor `(F ⋙ L, R) ⥤ (L, R)` -/
@[simps]
def preRight (L : A ⥤ T) (F : C ⥤ B) (R : B ⥤ T) : Comma L (F ⋙ R) ⥤ Comma L R where
  obj X :=
    { left := X.left
      right := F.obj X.right
      hom := X.hom }
  map f :=
    { left := f.left
      right := F.map f.right }


/-- `Comma.preRight` is a particular case of `Comma.map`,
but with better definitional properties. -/
def preRightIso (L : A ⥤ T) (F : C ⥤ B) (R : B ⥤ T) :
    preRight L F R ≅ map (L.leftUnitor.hom ≫ L.rightUnitor.inv) (F ⋙ R).rightUnitor.hom :=
                                /-
                                  A : Type u₁
                                  inst✝⁶ : CategoryTheory.Category.{v₁, u₁} A
                                  B : Type u₂
                                  inst✝⁵ : CategoryTheory.Category.{v₂, u₂} B
                                  T : Type u₃
                                  inst✝⁴ : CategoryTheory.Category.{v₃, u₃} T
                                  A' : Type u₄
                                  inst✝³ : CategoryTheory.Category.{v₄, u₄} A'
                                  B' : Type u₅
                                  inst✝² : CategoryTheory.Category.{v₅, u₅} B'
                                  T' : Type u₆
                                  inst✝¹ : CategoryTheory.Category.{v₆, u₆} T'
                                  L✝ : CategoryTheory.Functor A T
                                  R✝ : CategoryTheory.Functor B T
                                  C : Type u₄
                                  inst✝ : CategoryTheory.Category.{v₄, u₄} C
                                  L : CategoryTheory.Functor A T
                                  F : CategoryTheory.Functor C B
                                  R : CategoryTheory.Functor B T
                                  X : CategoryTheory.Comma L (F.comp R)
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map (CategoryTheory.Iso.refl ((Cat …
                                -/
                                /-
                                  🎉 no goals
                                -/
  NatIso.ofComponents (fun X => isoMk (Iso.refl _) (Iso.refl _))
  /-
    🎉 no goals
  -/


instance (L : A ⥤ T) (F : C ⥤ B) (R : B ⥤ T) [F.Faithful] : (preRight L F R).Faithful :=
  Functor.Faithful.of_iso (preRightIso L F R).symm


instance (L : A ⥤ T) (F : C ⥤ B) (R : B ⥤ T) [F.Full] : (preRight L F R).Full :=
  Functor.Full.of_iso (preRightIso L F R).symm


instance (L : A ⥤ T) (F : C ⥤ B) (R : B ⥤ T) [F.EssSurj] : (preRight L F R).EssSurj :=
  Functor.essSurj_of_iso (preRightIso L F R).symm


/-- If `F` is an equivalence, then so is `preRight L F R`. -/
instance isEquivalence_preRight (L : A ⥤ T) (F : C ⥤ B) (R : B ⥤ T) [F.IsEquivalence] :
    (preRight L F R).IsEquivalence where


/-- The functor `(L, R) ⥤ (L ⋙ F, R ⋙ F)` -/
@[simps]
def post (L : A ⥤ T) (R : B ⥤ T) (F : T ⥤ C) : Comma L R ⥤ Comma (L ⋙ F) (R ⋙ F) where
  obj X :=
    { left := X.left
      right := X.right
      hom := F.map X.hom }
  map f :=
    { left := f.left
      right := f.right
              /-
                A : Type u₁
                inst✝⁶ : CategoryTheory.Category.{v₁, u₁} A
                B : Type u₂
                inst✝⁵ : CategoryTheory.Category.{v₂, u₂} B
                T : Type u₃
                inst✝⁴ : CategoryTheory.Category.{v₃, u₃} T
                A' : Type u₄
                inst✝³ : CategoryTheory.Category.{v₄, u₄} A'
                B' : Type u₅
                inst✝² : CategoryTheory.Category.{v₅, u₅} B'
                T' : Type u₆
                inst✝¹ : CategoryTheory.Category.{v₆, u₆} T'
                L✝ : CategoryTheory.Functor A T
                R✝ : CategoryTheory.Functor B T
                C : Type u₄
                inst✝ : CategoryTheory.Category.{v₄, u₄} C
                L : CategoryTheory.Functor A T
                R : CategoryTheory.Functor B T
                F : CategoryTheory.Functor T C
                X✝ Y✝ : CategoryTheory.Comma L R
                f : Quiver.Hom X✝ Y✝
                ⊢ Eq (CategoryTheory.CategoryStruct.comp ((L.comp F).map f.left) ((fun X => {  …
              -/
      w := by simp only [Functor.comp_map, ← F.map_comp, f.w] }
              /-
                🎉 no goals
              -/


/-- `Comma.post` is a particular case of `Comma.map`, but with better definitional properties. -/
def postIso (L : A ⥤ T) (R : B ⥤ T) (F : T ⥤ C) :
    post L R F ≅ map (F₁ := 𝟭 _) (F₂ := 𝟭 _) (L ⋙ F).leftUnitor.hom (R ⋙ F).leftUnitor.inv :=
                                /-
                                  A : Type u₁
                                  inst✝⁶ : CategoryTheory.Category.{v₁, u₁} A
                                  B : Type u₂
                                  inst✝⁵ : CategoryTheory.Category.{v₂, u₂} B
                                  T : Type u₃
                                  inst✝⁴ : CategoryTheory.Category.{v₃, u₃} T
                                  A' : Type u₄
                                  inst✝³ : CategoryTheory.Category.{v₄, u₄} A'
                                  B' : Type u₅
                                  inst✝² : CategoryTheory.Category.{v₅, u₅} B'
                                  T' : Type u₆
                                  inst✝¹ : CategoryTheory.Category.{v₆, u₆} T'
                                  L✝ : CategoryTheory.Functor A T
                                  R✝ : CategoryTheory.Functor B T
                                  C : Type u₄
                                  inst✝ : CategoryTheory.Category.{v₄, u₄} C
                                  L : CategoryTheory.Functor A T
                                  R : CategoryTheory.Functor B T
                                  F : CategoryTheory.Functor T C
                                  X : CategoryTheory.Comma L R
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp ((L.comp F).map (CategoryTheory.Iso.r …
                                -/
                                /-
                                  🎉 no goals
                                -/
  NatIso.ofComponents (fun X => isoMk (Iso.refl _) (Iso.refl _))
  /-
    🎉 no goals
  -/


instance (L : A ⥤ T) (R : B ⥤ T) (F : T ⥤ C) : (post L R F).Faithful :=
  Functor.Faithful.of_iso (postIso L R F).symm


instance (L : A ⥤ T) (R : B ⥤ T) (F : T ⥤ C) [F.Faithful] : (post L R F).Full :=
  Functor.Full.of_iso (postIso L R F).symm


instance (L : A ⥤ T) (R : B ⥤ T) (F : T ⥤ C) [F.Full] : (post L R F).EssSurj :=
  Functor.essSurj_of_iso (postIso L R F).symm


/-- If `F` is an equivalence, then so is `post L R F`. -/
instance isEquivalence_post (L : A ⥤ T) (R : B ⥤ T) (F : T ⥤ C) [F.IsEquivalence] :
    (post L R F).IsEquivalence where


/-- The canonical functor from the product of two categories to the comma category of their
respective functors into `Discrete PUnit`. -/
@[simps]
def fromProd (L : A ⥤ Discrete PUnit) (R : B ⥤ Discrete PUnit) :
    A × B ⥤ Comma L R where
  obj X :=
    { left := X.1
      right := X.2
      hom := Discrete.eqToHom rfl }
  map {X} {Y} f :=
    { left := f.1
      right := f.2 }


/-- Taking the comma category of two functors into `Discrete PUnit` results in something
is equivalent to their product. -/
@[simps!]
def equivProd (L : A ⥤ Discrete PUnit) (R : B ⥤ Discrete PUnit) :
    Comma L R ≌ A × B where
  functor := (fst L R).prod' (snd L R)
  inverse := fromProd L R
  unitIso := Iso.refl _
  counitIso := Iso.refl _


/-- Taking the comma category of a functor into `A ⥤ Discrete PUnit` and the identity
`Discrete PUnit ⥤ Discrete PUnit` results in a category equivalent to `A`. -/
@[simps!]
def toPUnitIdEquiv (L : A ⥤ Discrete PUnit) (R : Discrete PUnit ⥤ Discrete PUnit) :
    Comma L R ≌ A :=
  (equivProd L _).trans (prod.rightUnitorEquivalence A)


@[simp]
theorem toPUnitIdEquiv_functor_iso {L : A ⥤ Discrete PUnit}
    {R : Discrete PUnit ⥤ Discrete PUnit} :
    (toPUnitIdEquiv L R).functor = fst L R :=
  rfl


/-- Taking the comma category of the identity `Discrete PUnit ⥤ Discrete PUnit`
and a functor `B ⥤ Discrete PUnit` results in a category equivalent to `B`. -/
@[simps!]
def toIdPUnitEquiv (L : Discrete PUnit ⥤ Discrete PUnit) (R : B ⥤ Discrete PUnit) :
    Comma L R ≌ B :=
  (equivProd _ R).trans (prod.leftUnitorEquivalence B)


@[simp]
theorem toIdPUnitEquiv_functor_iso {L : Discrete PUnit ⥤ Discrete PUnit}
    {R : B ⥤ Discrete PUnit} :
    (toIdPUnitEquiv L R).functor = snd L R :=
  rfl


/-- The canonical functor from `Comma L R` to `(Comma R.op L.op)ᵒᵖ`. -/
@[simps]
def opFunctor : Comma L R ⥤ (Comma R.op L.op)ᵒᵖ where
  obj X := ⟨op X.right, op X.left, op X.hom⟩
                                                           /-
                                                             A : Type u₁
                                                             inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
                                                             B : Type u₂
                                                             inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
                                                             T : Type u₃
                                                             inst✝³ : CategoryTheory.Category.{v₃, u₃} T
                                                             A' : Type u₄
                                                             inst✝² : CategoryTheory.Category.{v₄, u₄} A'
                                                             B' : Type u₅
                                                             inst✝¹ : CategoryTheory.Category.{v₅, u₅} B'
                                                             T' : Type u₆
                                                             inst✝ : CategoryTheory.Category.{v₆, u₆} T'
                                                             L : CategoryTheory.Functor A T
                                                             R : CategoryTheory.Functor B T
                                                             X✝ Y✝ : CategoryTheory.Comma L R
                                                             f : Quiver.Hom X✝ Y✝
                                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (R.op.map { unop := f.right }) (Oppos …
                                                           -/
  map f := ⟨op f.right, op f.left, Quiver.Hom.unop_inj (by simp)⟩
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- Composing the `leftOp` of `opFunctor L R` with `fst L.op R.op` is naturally isomorphic
to `snd L R`.-/
@[simps!]
def opFunctorCompFst : (opFunctor L R).leftOp ⋙ fst _ _ ≅ (snd _ _).op :=
  Iso.refl _


/-- Composing the `leftOp` of `opFunctor L R` with `snd L.op R.op` is naturally isomorphic
to `fst L R`.-/
@[simps!]
def opFunctorCompSnd : (opFunctor L R).leftOp ⋙ snd _ _ ≅ (fst _ _).op :=
  Iso.refl _


/-- The canonical functor from `Comma L.op R.op` to `(Comma R L)ᵒᵖ`. -/
@[simps]
def unopFunctor : Comma L.op R.op ⥤ (Comma R L)ᵒᵖ where
  obj X := ⟨X.right.unop, X.left.unop, X.hom.unop⟩
                                                             /-
                                                               A : Type u₁
                                                               inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
                                                               B : Type u₂
                                                               inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
                                                               T : Type u₃
                                                               inst✝³ : CategoryTheory.Category.{v₃, u₃} T
                                                               A' : Type u₄
                                                               inst✝² : CategoryTheory.Category.{v₄, u₄} A'
                                                               B' : Type u₅
                                                               inst✝¹ : CategoryTheory.Category.{v₅, u₅} B'
                                                               T' : Type u₆
                                                               inst✝ : CategoryTheory.Category.{v₆, u₆} T'
                                                               L : CategoryTheory.Functor A T
                                                               R : CategoryTheory.Functor B T
                                                               X✝ Y✝ : CategoryTheory.Comma L.op R.op
                                                               f : Quiver.Hom X✝ Y✝
                                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (R.map f.right.unop) (Opposite.unop ( …
                                                             -/
  map f := ⟨f.right.unop, f.left.unop, Quiver.Hom.op_inj (by simpa using f.w.symm)⟩
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- Composing `unopFunctor L R` with `(fst L R).op` is isomorphic to `snd L.op R.op`. -/
@[simps!]
def unopFunctorCompFst : unopFunctor L R ⋙ (fst _ _).op ≅ snd _ _ :=
  Iso.refl _


/-- Composing `unopFunctor L R` with `(snd L R).op` is isomorphic to `fst L.op R.op`. -/
@[simps!]
def unopFunctorCompSnd : unopFunctor L R ⋙ (snd _ _).op ≅ fst _ _ :=
  Iso.refl _


/-- The canonical equivalence between `Comma L R` and `(Comma R.op L.op)ᵒᵖ`. -/
@[simps]
def opEquiv : Comma L R ≌ (Comma R.op L.op)ᵒᵖ where
  functor := opFunctor L R
  inverse := (unopFunctor R L).leftOp
             /-
               A : Type u₁
               inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
               B : Type u₂
               inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
               T : Type u₃
               inst✝³ : CategoryTheory.Category.{v₃, u₃} T
               A' : Type u₄
               inst✝² : CategoryTheory.Category.{v₄, u₄} A'
               B' : Type u₅
               inst✝¹ : CategoryTheory.Category.{v₅, u₅} B'
               T' : Type u₆
               inst✝ : CategoryTheory.Category.{v₆, u₆} T'
               L : CategoryTheory.Functor A T
               R : CategoryTheory.Functor B T
               ⊢ ∀ {X Y : CategoryTheory.Comma L R} (f : Quiver.Hom X Y), Eq (CategoryTheory. …
             -/
  unitIso := NatIso.ofComponents (fun X => Iso.refl _)
             /-
               🎉 no goals
             -/
               /-
                 A : Type u₁
                 inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
                 B : Type u₂
                 inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
                 T : Type u₃
                 inst✝³ : CategoryTheory.Category.{v₃, u₃} T
                 A' : Type u₄
                 inst✝² : CategoryTheory.Category.{v₄, u₄} A'
                 B' : Type u₅
                 inst✝¹ : CategoryTheory.Category.{v₅, u₅} B'
                 T' : Type u₆
                 inst✝ : CategoryTheory.Category.{v₆, u₆} T'
                 L : CategoryTheory.Functor A T
                 R : CategoryTheory.Functor B T
                 ⊢ ∀ {X Y : Opposite (CategoryTheory.Comma R.op L.op)} (f : Quiver.Hom X Y), Eq …
               -/
  counitIso := NatIso.ofComponents (fun X => Iso.refl _)
               /-
                 🎉 no goals
               -/


