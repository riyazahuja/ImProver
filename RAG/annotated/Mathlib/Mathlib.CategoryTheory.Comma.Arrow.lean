/-- The arrow category of `T` has as objects all morphisms in `T` and as morphisms commutative
     squares in `T`. -/
def Arrow :=
  Comma.{v, v, v} (𝟭 T) (𝟭 T)

/- Porting note: could not derive `Category` above so this instance works in its place -/

instance : Category (Arrow T) := commaCategory

-- Satisfying the inhabited linter

instance Arrow.inhabited [Inhabited T] : Inhabited (Arrow T) where
  default := show Comma (𝟭 T) (𝟭 T) from default


@[ext]
lemma hom_ext {X Y : Arrow T} (f g : X ⟶ Y) (h₁ : f.left = g.left) (h₂ : f.right = g.right) :
    f = g :=
  CommaMorphism.ext h₁ h₂


@[simp]
theorem id_left (f : Arrow T) : CommaMorphism.left (𝟙 f) = 𝟙 f.left :=
  rfl


@[simp]
theorem id_right (f : Arrow T) : CommaMorphism.right (𝟙 f) = 𝟙 f.right :=
  rfl

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10688): added to ease automation

@[simp, reassoc]
theorem comp_left {X Y Z : Arrow T} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).left = f.left ≫ g.left := rfl

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10688): added to ease automation

@[simp, reassoc]
theorem comp_right {X Y Z : Arrow T} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).right = f.right ≫ g.right := rfl


/-- An object in the arrow category is simply a morphism in `T`. -/
@[simps]
def mk {X Y : T} (f : X ⟶ Y) : Arrow T where
  left := X
  right := Y
  hom := f


@[simp]
theorem mk_eq (f : Arrow T) : Arrow.mk f.hom = f := by
  /-
    T : Type u
    inst✝ : CategoryTheory.Category.{v, u} T
    f : CategoryTheory.Arrow T
    ⊢ Eq (CategoryTheory.Arrow.mk f.hom) f
  -/
  cases f
  /-
    case mk
    T : Type u
    inst✝ : CategoryTheory.Category.{v, u} T
    left✝ right✝ : T
    hom✝ : Quiver.Hom ((CategoryTheory.Functor.id T).obj left✝) ((CategoryTheory.F …
    ⊢ Eq (CategoryTheory.Arrow.mk { left := left✝, right := right✝, hom := hom✝ }. …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem mk_injective (A B : T) :
    Function.Injective (Arrow.mk : (A ⟶ B) → Arrow T) := fun f g h => by
  /-
    T : Type u
    inst✝ : CategoryTheory.Category.{v, u} T
    A B : T
    f g : Quiver.Hom A B
    h : Eq (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk g)
    ⊢ Eq f g
  -/
  cases h
  /-
    case refl
    T : Type u
    inst✝ : CategoryTheory.Category.{v, u} T
    A B : T
    f : Quiver.Hom A B
    ⊢ Eq f f
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem mk_inj (A B : T) {f g : A ⟶ B} : Arrow.mk f = Arrow.mk g ↔ f = g :=
  (mk_injective A B).eq_iff


instance {X Y : T} : CoeOut (X ⟶ Y) (Arrow T) where
  coe := mk


/-- A morphism in the arrow category is a commutative square connecting two objects of the arrow
    category. -/
@[simps]
def homMk {f g : Arrow T} {u : f.left ⟶ g.left} {v : f.right ⟶ g.right}
    (w : u ≫ g.hom = f.hom ≫ v) : f ⟶ g where
  left := u
  right := v
  w := w


/-- We can also build a morphism in the arrow category out of any commutative square in `T`. -/
@[simps]
def homMk' {X Y : T} {f : X ⟶ Y} {P Q : T} {g : P ⟶ Q} {u : X ⟶ P} {v : Y ⟶ Q} (w : u ≫ g = f ≫ v) :
    Arrow.mk f ⟶ Arrow.mk g where
  left := u
  right := v
  w := w

/- Porting note: was warned simp could prove reassoc'd version. Found simp could not.
Added nolint. -/

@[reassoc (attr := simp, nolint simpNF)]
theorem w {f g : Arrow T} (sq : f ⟶ g) : sq.left ≫ g.hom = f.hom ≫ sq.right :=
  sq.w

-- `w_mk_left` is not needed, as it is a consequence of `w` and `mk_hom`.

@[reassoc (attr := simp)]
theorem w_mk_right {f : Arrow T} {X Y : T} {g : X ⟶ Y} (sq : f ⟶ mk g) :
    sq.left ≫ g = f.hom ≫ sq.right :=
  sq.w


theorem isIso_of_isIso_left_of_isIso_right {f g : Arrow T} (ff : f ⟶ g) [IsIso ff.left]
    [IsIso ff.right] : IsIso ff where
  out := by
    /-
      T : Type u
      inst✝² : CategoryTheory.Category.{v, u} T
      f g : CategoryTheory.Arrow T
      ff : Quiver.Hom f g
      inst✝¹ : CategoryTheory.IsIso ff.left
      inst✝ : CategoryTheory.IsIso ff.right
      ⊢ Exists fun inv => And (Eq (CategoryTheory.CategoryStruct.comp ff inv) (Categ …
    -/
    let inverse : g ⟶ f := ⟨inv ff.left, inv ff.right, (by simp)⟩
    /-
      T : Type u
      inst✝² : CategoryTheory.Category.{v, u} T
      f g : CategoryTheory.Arrow T
      ff : Quiver.Hom f g
      inst✝¹ : CategoryTheory.IsIso ff.left
      inst✝ : CategoryTheory.IsIso ff.right
      inverse : Quiver.Hom g f := { left := CategoryTheory.inv ff.left, right := Cat …
      ⊢ Exists fun inv => And (Eq (CategoryTheory.CategoryStruct.comp ff inv) (Categ …
    -/
    apply Exists.intro inverse
    /-
      T : Type u
      inst✝² : CategoryTheory.Category.{v, u} T
      f g : CategoryTheory.Arrow T
      ff : Quiver.Hom f g
      inst✝¹ : CategoryTheory.IsIso ff.left
      inst✝ : CategoryTheory.IsIso ff.right
      inverse : Quiver.Hom g f := { left := CategoryTheory.inv ff.left, right := Cat …
      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp ff inverse) (CategoryTheory.Cate …
    -/
    aesop_cat
    /-
      🎉 no goals
    -/


/-- Create an isomorphism between arrows,
by providing isomorphisms between the domains and codomains,
and a proof that the square commutes. -/
@[simps!]
def isoMk {f g : Arrow T} (l : f.left ≅ g.left) (r : f.right ≅ g.right)
    (h : l.hom ≫ g.hom = f.hom ≫ r.hom := by aesop_cat) : f ≅ g :=
  Comma.isoMk l r h


/-- A variant of `Arrow.isoMk` that creates an iso between two `Arrow.mk`s with a better type
signature. -/
abbrev isoMk' {W X Y Z : T} (f : W ⟶ X) (g : Y ⟶ Z) (e₁ : W ≅ Y) (e₂ : X ≅ Z)
    (h : e₁.hom ≫ g = f ≫ e₂.hom := by aesop_cat) : Arrow.mk f ≅ Arrow.mk g :=
  Arrow.isoMk e₁ e₂ h


theorem hom.congr_left {f g : Arrow T} {φ₁ φ₂ : f ⟶ g} (h : φ₁ = φ₂) : φ₁.left = φ₂.left := by
  /-
    T : Type u
    inst✝ : CategoryTheory.Category.{v, u} T
    f g : CategoryTheory.Arrow T
    φ₁ φ₂ : Quiver.Hom f g
    h : Eq φ₁ φ₂
    ⊢ Eq φ₁.left φ₂.left
  -/
  rw [h]
  /-
    🎉 no goals
  -/


@[simp]
theorem hom.congr_right {f g : Arrow T} {φ₁ φ₂ : f ⟶ g} (h : φ₁ = φ₂) : φ₁.right = φ₂.right := by
  /-
    T : Type u
    inst✝ : CategoryTheory.Category.{v, u} T
    f g : CategoryTheory.Arrow T
    φ₁ φ₂ : Quiver.Hom f g
    h : Eq φ₁ φ₂
    ⊢ Eq φ₁.right φ₂.right
  -/
  rw [h]
  /-
    🎉 no goals
  -/


theorem iso_w {f g : Arrow T} (e : f ≅ g) : g.hom = e.inv.left ≫ f.hom ≫ e.hom.right := by
  /-
    T : Type u
    inst✝ : CategoryTheory.Category.{v, u} T
    f g : CategoryTheory.Arrow T
    e : CategoryTheory.Iso f g
    ⊢ Eq g.hom (CategoryTheory.CategoryStruct.comp e.inv.left (CategoryTheory.Cate …
  -/
  have eq := Arrow.hom.congr_right e.inv_hom_id
  /-
    T : Type u
    inst✝ : CategoryTheory.Category.{v, u} T
    f g : CategoryTheory.Arrow T
    e : CategoryTheory.Iso f g
    eq : Eq (CategoryTheory.CategoryStruct.comp e.inv e.hom).right (CategoryTheory …
    ⊢ Eq g.hom (CategoryTheory.CategoryStruct.comp e.inv.left (CategoryTheory.Cate …
  -/
  rw [Arrow.comp_right, Arrow.id_right] at eq
  /-
    T : Type u
    inst✝ : CategoryTheory.Category.{v, u} T
    f g : CategoryTheory.Arrow T
    e : CategoryTheory.Iso f g
    eq : Eq (CategoryTheory.CategoryStruct.comp e.inv.right e.hom.right) (Category …
    ⊢ Eq g.hom (CategoryTheory.CategoryStruct.comp e.inv.left (CategoryTheory.Cate …
  -/
  rw [Arrow.w_assoc, eq, Category.comp_id]
  /-
    🎉 no goals
  -/


theorem iso_w' {W X Y Z : T} {f : W ⟶ X} {g : Y ⟶ Z} (e : Arrow.mk f ≅ Arrow.mk g) :
    g = e.inv.left ≫ f ≫ e.hom.right :=
  iso_w e


instance isIso_left [IsIso sq] : IsIso sq.left where
  out := by
    /-
      T : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} T
      f g : CategoryTheory.Arrow T
      sq : Quiver.Hom f g
      inst✝ : CategoryTheory.IsIso sq
      ⊢ Exists fun inv => And (Eq (CategoryTheory.CategoryStruct.comp sq.left inv) ( …
    -/
    apply Exists.intro (inv sq).left
    simp only [← Comma.comp_left, IsIso.hom_inv_id, IsIso.inv_hom_id, Arrow.id_left,
      eq_self_iff_true, and_self_iff]
    /-
      T : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} T
      f g : CategoryTheory.Arrow T
      sq : Quiver.Hom f g
      inst✝ : CategoryTheory.IsIso sq
      ⊢ And (Eq (CategoryTheory.CategoryStruct.id f).left (CategoryTheory.CategorySt …
    -/
    simp
    /-
      🎉 no goals
    -/


instance isIso_right [IsIso sq] : IsIso sq.right where
  out := by
    /-
      T : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} T
      f g : CategoryTheory.Arrow T
      sq : Quiver.Hom f g
      inst✝ : CategoryTheory.IsIso sq
      ⊢ Exists fun inv => And (Eq (CategoryTheory.CategoryStruct.comp sq.right inv)  …
    -/
    apply Exists.intro (inv sq).right
    simp only [← Comma.comp_right, IsIso.hom_inv_id, IsIso.inv_hom_id, Arrow.id_right,
      eq_self_iff_true, and_self_iff]
    /-
      T : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} T
      f g : CategoryTheory.Arrow T
      sq : Quiver.Hom f g
      inst✝ : CategoryTheory.IsIso sq
      ⊢ And (Eq (CategoryTheory.CategoryStruct.id f).right (CategoryTheory.CategoryS …
    -/
    simp
    /-
      🎉 no goals
    -/


@[simp]
theorem inv_left [IsIso sq] : (inv sq).left = inv sq.left :=
                                   /-
                                     T : Type u
                                     inst✝¹ : CategoryTheory.Category.{v, u} T
                                     f g : CategoryTheory.Arrow T
                                     sq : Quiver.Hom f g
                                     inst✝ : CategoryTheory.IsIso sq
                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp sq.left (CategoryTheory.inv sq).left) …
                                   -/
  IsIso.eq_inv_of_hom_inv_id <| by rw [← Comma.comp_left, IsIso.hom_inv_id, id_left]
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
theorem inv_right [IsIso sq] : (inv sq).right = inv sq.right :=
                                   /-
                                     T : Type u
                                     inst✝¹ : CategoryTheory.Category.{v, u} T
                                     f g : CategoryTheory.Arrow T
                                     sq : Quiver.Hom f g
                                     inst✝ : CategoryTheory.IsIso sq
                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp sq.right (CategoryTheory.inv sq).righ …
                                   -/
  IsIso.eq_inv_of_hom_inv_id <| by rw [← Comma.comp_right, IsIso.hom_inv_id, id_right]
                                   /-
                                     🎉 no goals
                                   -/


theorem left_hom_inv_right [IsIso sq] : sq.left ≫ g.hom ≫ inv sq.right = f.hom := by
  /-
    T : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} T
    f g : CategoryTheory.Arrow T
    sq : Quiver.Hom f g
    inst✝ : CategoryTheory.IsIso sq
    ⊢ Eq (CategoryTheory.CategoryStruct.comp sq.left (CategoryTheory.CategoryStruc …
  -/
  simp only [← Category.assoc, IsIso.comp_inv_eq, w]
  /-
    🎉 no goals
  -/


theorem inv_left_hom_right [IsIso sq] : inv sq.left ≫ f.hom ≫ sq.right = g.hom := by
  /-
    T : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} T
    f g : CategoryTheory.Arrow T
    sq : Quiver.Hom f g
    inst✝ : CategoryTheory.IsIso sq
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv sq.left) (Categor …
  -/
  simp only [w, IsIso.inv_comp_eq]
  /-
    🎉 no goals
  -/


instance mono_left [Mono sq] : Mono sq.left where
  right_cancellation {Z} φ ψ h := by
    let aux : (Z ⟶ f.left) → (Arrow.mk (𝟙 Z) ⟶ f) := fun φ =>
      { left := φ
        right := φ ≫ f.hom }
    /-
      T : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} T
      f g : CategoryTheory.Arrow T
      sq : Quiver.Hom f g
      inst✝ : CategoryTheory.Mono sq
      Z : T
      φ ψ : Quiver.Hom Z f.left
      h : Eq (CategoryTheory.CategoryStruct.comp φ sq.left) (CategoryTheory.Category …
      aux : Quiver.Hom Z f.left → Quiver.Hom (CategoryTheory.Arrow.mk (CategoryTheor …
      ⊢ Eq φ ψ
    -/
    have : ∀ g, (aux g).right = g ≫ f.hom := fun g => by dsimp
    /-
      T : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} T
      f g : CategoryTheory.Arrow T
      sq : Quiver.Hom f g
      inst✝ : CategoryTheory.Mono sq
      Z : T
      φ ψ : Quiver.Hom Z f.left
      h : Eq (CategoryTheory.CategoryStruct.comp φ sq.left) (CategoryTheory.Category …
      aux : Quiver.Hom Z f.left → Quiver.Hom (CategoryTheory.Arrow.mk (CategoryTheor …
      this : ∀ (g : Quiver.Hom Z f.left), Eq (aux g).right (CategoryTheory.CategoryS …
      ⊢ Eq φ ψ
    -/
    show (aux φ).left = (aux ψ).left
    /-
      T : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} T
      f g : CategoryTheory.Arrow T
      sq : Quiver.Hom f g
      inst✝ : CategoryTheory.Mono sq
      Z : T
      φ ψ : Quiver.Hom Z f.left
      h : Eq (CategoryTheory.CategoryStruct.comp φ sq.left) (CategoryTheory.Category …
      aux : Quiver.Hom Z f.left → Quiver.Hom (CategoryTheory.Arrow.mk (CategoryTheor …
      this : ∀ (g : Quiver.Hom Z f.left), Eq (aux g).right (CategoryTheory.CategoryS …
      ⊢ Eq (aux φ).left (aux ψ).left
    -/
    congr 1
    /-
      case e_self
      T : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} T
      f g : CategoryTheory.Arrow T
      sq : Quiver.Hom f g
      inst✝ : CategoryTheory.Mono sq
      Z : T
      φ ψ : Quiver.Hom Z f.left
      h : Eq (CategoryTheory.CategoryStruct.comp φ sq.left) (CategoryTheory.Category …
      aux : Quiver.Hom Z f.left → Quiver.Hom (CategoryTheory.Arrow.mk (CategoryTheor …
      this : ∀ (g : Quiver.Hom Z f.left), Eq (aux g).right (CategoryTheory.CategoryS …
      ⊢ Eq (aux φ) (aux ψ)
    -/
    rw [← cancel_mono sq]
    /-
      case e_self
      T : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} T
      f g : CategoryTheory.Arrow T
      sq : Quiver.Hom f g
      inst✝ : CategoryTheory.Mono sq
      Z : T
      φ ψ : Quiver.Hom Z f.left
      h : Eq (CategoryTheory.CategoryStruct.comp φ sq.left) (CategoryTheory.Category …
      aux : Quiver.Hom Z f.left → Quiver.Hom (CategoryTheory.Arrow.mk (CategoryTheor …
      this : ∀ (g : Quiver.Hom Z f.left), Eq (aux g).right (CategoryTheory.CategoryS …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (aux φ) sq) (CategoryTheory.CategoryS …
    -/
    apply CommaMorphism.ext
      /-
        case e_self.left
        T : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} T
        f g : CategoryTheory.Arrow T
        sq : Quiver.Hom f g
        inst✝ : CategoryTheory.Mono sq
        Z : T
        φ ψ : Quiver.Hom Z f.left
        h : Eq (CategoryTheory.CategoryStruct.comp φ sq.left) (CategoryTheory.Category …
        aux : Quiver.Hom Z f.left → Quiver.Hom (CategoryTheory.Arrow.mk (CategoryTheor …
        this : ∀ (g : Quiver.Hom Z f.left), Eq (aux g).right (CategoryTheory.CategoryS …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (aux φ) sq).left (CategoryTheory.Cate …
      -/
    · exact h
      /-
        🎉 no goals
      -/
      /-
        case e_self.right
        T : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} T
        f g : CategoryTheory.Arrow T
        sq : Quiver.Hom f g
        inst✝ : CategoryTheory.Mono sq
        Z : T
        φ ψ : Quiver.Hom Z f.left
        h : Eq (CategoryTheory.CategoryStruct.comp φ sq.left) (CategoryTheory.Category …
        aux : Quiver.Hom Z f.left → Quiver.Hom (CategoryTheory.Arrow.mk (CategoryTheor …
        this : ∀ (g : Quiver.Hom Z f.left), Eq (aux g).right (CategoryTheory.CategoryS …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (aux φ) sq).right (CategoryTheory.Cat …
      -/
    · rw [Comma.comp_right, Comma.comp_right, this, this, Category.assoc, Category.assoc]
      /-
        case e_self.right
        T : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} T
        f g : CategoryTheory.Arrow T
        sq : Quiver.Hom f g
        inst✝ : CategoryTheory.Mono sq
        Z : T
        φ ψ : Quiver.Hom Z f.left
        h : Eq (CategoryTheory.CategoryStruct.comp φ sq.left) (CategoryTheory.Category …
        aux : Quiver.Hom Z f.left → Quiver.Hom (CategoryTheory.Arrow.mk (CategoryTheor …
        this : ∀ (g : Quiver.Hom Z f.left), Eq (aux g).right (CategoryTheory.CategoryS …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (CategoryTheory.CategoryStruct.comp …
      -/
      rw [← Arrow.w]
      /-
        case e_self.right
        T : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} T
        f g : CategoryTheory.Arrow T
        sq : Quiver.Hom f g
        inst✝ : CategoryTheory.Mono sq
        Z : T
        φ ψ : Quiver.Hom Z f.left
        h : Eq (CategoryTheory.CategoryStruct.comp φ sq.left) (CategoryTheory.Category …
        aux : Quiver.Hom Z f.left → Quiver.Hom (CategoryTheory.Arrow.mk (CategoryTheor …
        this : ∀ (g : Quiver.Hom Z f.left), Eq (aux g).right (CategoryTheory.CategoryS …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (CategoryTheory.CategoryStruct.comp …
      -/
      simp only [← Category.assoc, h]
      /-
        🎉 no goals
      -/


instance epi_right [Epi sq] : Epi sq.right where
  left_cancellation {Z} φ ψ h := by
    let aux : (g.right ⟶ Z) → (g ⟶ Arrow.mk (𝟙 Z)) := fun φ =>
      { right := φ
        left := g.hom ≫ φ }
    /-
      T : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} T
      f g : CategoryTheory.Arrow T
      sq : Quiver.Hom f g
      inst✝ : CategoryTheory.Epi sq
      Z : T
      φ ψ : Quiver.Hom g.right Z
      h : Eq (CategoryTheory.CategoryStruct.comp sq.right φ) (CategoryTheory.Categor …
      aux : Quiver.Hom g.right Z → Quiver.Hom g (CategoryTheory.Arrow.mk (CategoryTh …
      ⊢ Eq φ ψ
    -/
    show (aux φ).right = (aux ψ).right
    /-
      T : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} T
      f g : CategoryTheory.Arrow T
      sq : Quiver.Hom f g
      inst✝ : CategoryTheory.Epi sq
      Z : T
      φ ψ : Quiver.Hom g.right Z
      h : Eq (CategoryTheory.CategoryStruct.comp sq.right φ) (CategoryTheory.Categor …
      aux : Quiver.Hom g.right Z → Quiver.Hom g (CategoryTheory.Arrow.mk (CategoryTh …
      ⊢ Eq (aux φ).right (aux ψ).right
    -/
    congr 1
    /-
      case e_self
      T : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} T
      f g : CategoryTheory.Arrow T
      sq : Quiver.Hom f g
      inst✝ : CategoryTheory.Epi sq
      Z : T
      φ ψ : Quiver.Hom g.right Z
      h : Eq (CategoryTheory.CategoryStruct.comp sq.right φ) (CategoryTheory.Categor …
      aux : Quiver.Hom g.right Z → Quiver.Hom g (CategoryTheory.Arrow.mk (CategoryTh …
      ⊢ Eq (aux φ) (aux ψ)
    -/
    rw [← cancel_epi sq]
    /-
      case e_self
      T : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} T
      f g : CategoryTheory.Arrow T
      sq : Quiver.Hom f g
      inst✝ : CategoryTheory.Epi sq
      Z : T
      φ ψ : Quiver.Hom g.right Z
      h : Eq (CategoryTheory.CategoryStruct.comp sq.right φ) (CategoryTheory.Categor …
      aux : Quiver.Hom g.right Z → Quiver.Hom g (CategoryTheory.Arrow.mk (CategoryTh …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp sq (aux φ)) (CategoryTheory.CategoryS …
    -/
    apply CommaMorphism.ext
      /-
        case e_self.left
        T : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} T
        f g : CategoryTheory.Arrow T
        sq : Quiver.Hom f g
        inst✝ : CategoryTheory.Epi sq
        Z : T
        φ ψ : Quiver.Hom g.right Z
        h : Eq (CategoryTheory.CategoryStruct.comp sq.right φ) (CategoryTheory.Categor …
        aux : Quiver.Hom g.right Z → Quiver.Hom g (CategoryTheory.Arrow.mk (CategoryTh …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp sq (aux φ)).left (CategoryTheory.Cate …
      -/
    · rw [Comma.comp_left, Comma.comp_left, Arrow.w_assoc, Arrow.w_assoc, h]
      /-
        🎉 no goals
      -/
      /-
        case e_self.right
        T : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} T
        f g : CategoryTheory.Arrow T
        sq : Quiver.Hom f g
        inst✝ : CategoryTheory.Epi sq
        Z : T
        φ ψ : Quiver.Hom g.right Z
        h : Eq (CategoryTheory.CategoryStruct.comp sq.right φ) (CategoryTheory.Categor …
        aux : Quiver.Hom g.right Z → Quiver.Hom g (CategoryTheory.Arrow.mk (CategoryTh …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp sq (aux φ)).right (CategoryTheory.Cat …
      -/
    · exact h
      /-
        🎉 no goals
      -/


@[reassoc (attr := simp)]
lemma hom_inv_id_left (e : f ≅ g) : e.hom.left ≫ e.inv.left = 𝟙 _ := by
  /-
    T : Type u
    inst✝ : CategoryTheory.Category.{v, u} T
    f g : CategoryTheory.Arrow T
    e : CategoryTheory.Iso f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp e.hom.left e.inv.left) (CategoryTheor …
  -/
  rw [← comp_left, e.hom_inv_id, id_left]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma inv_hom_id_left (e : f ≅ g) : e.inv.left ≫ e.hom.left = 𝟙 _ := by
  /-
    T : Type u
    inst✝ : CategoryTheory.Category.{v, u} T
    f g : CategoryTheory.Arrow T
    e : CategoryTheory.Iso f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp e.inv.left e.hom.left) (CategoryTheor …
  -/
  rw [← comp_left, e.inv_hom_id, id_left]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma hom_inv_id_right (e : f ≅ g) : e.hom.right ≫ e.inv.right = 𝟙 _ := by
  /-
    T : Type u
    inst✝ : CategoryTheory.Category.{v, u} T
    f g : CategoryTheory.Arrow T
    e : CategoryTheory.Iso f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp e.hom.right e.inv.right) (CategoryThe …
  -/
  rw [← comp_right, e.hom_inv_id, id_right]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma inv_hom_id_right (e : f ≅ g) : e.inv.right ≫ e.hom.right = 𝟙 _ := by
  /-
    T : Type u
    inst✝ : CategoryTheory.Category.{v, u} T
    f g : CategoryTheory.Arrow T
    e : CategoryTheory.Iso f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp e.inv.right e.hom.right) (CategoryThe …
  -/
  rw [← comp_right, e.inv_hom_id, id_right]
  /-
    🎉 no goals
  -/


/-- Given a square from an arrow `i` to an isomorphism `p`, express the source part of `sq`
in terms of the inverse of `p`. -/
@[simp]
theorem square_to_iso_invert (i : Arrow T) {X Y : T} (p : X ≅ Y) (sq : i ⟶ Arrow.mk p.hom) :
    i.hom ≫ sq.right ≫ p.inv = sq.left := by
  /-
    T : Type u
    inst✝ : CategoryTheory.Category.{v, u} T
    i : CategoryTheory.Arrow T
    X Y : T
    p : CategoryTheory.Iso X Y
    sq : Quiver.Hom i (CategoryTheory.Arrow.mk p.hom)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp i.hom (CategoryTheory.CategoryStruct. …
  -/
  simpa only [Category.assoc] using (Iso.comp_inv_eq p).mpr (Arrow.w_mk_right sq).symm
  /-
    🎉 no goals
  -/


/-- Given a square from an isomorphism `i` to an arrow `p`, express the target part of `sq`
in terms of the inverse of `i`. -/
theorem square_from_iso_invert {X Y : T} (i : X ≅ Y) (p : Arrow T) (sq : Arrow.mk i.hom ⟶ p) :
                                             /-
                                               T : Type u
                                               inst✝ : CategoryTheory.Category.{v, u} T
                                               X Y : T
                                               i : CategoryTheory.Iso X Y
                                               p : CategoryTheory.Arrow T
                                               sq : Quiver.Hom (CategoryTheory.Arrow.mk i.hom) p
                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp i.inv (CategoryTheory.CategoryStruct. …
                                             -/
    i.inv ≫ sq.left ≫ p.hom = sq.right := by simp only [Iso.inv_hom_id_assoc, Arrow.w, Arrow.mk_hom]
                                             /-
                                               🎉 no goals
                                             -/


/-- A helper construction: given a square between `i` and `f ≫ g`, produce a square between
`i` and `g`, whose top leg uses `f`:
A  → X
     ↓f
↓i   Y             --> A → Y
     ↓g                ↓i  ↓g
B  → Z                 B → Z
 -/
@[simps]
def squareToSnd {X Y Z : C} {i : Arrow C} {f : X ⟶ Y} {g : Y ⟶ Z} (sq : i ⟶ Arrow.mk (f ≫ g)) :
    i ⟶ Arrow.mk g where
  left := sq.left ≫ f
  right := sq.right


/-- The functor sending an arrow to its source. -/
@[simps!]
def leftFunc : Arrow C ⥤ C :=
  Comma.fst _ _


/-- The functor sending an arrow to its target. -/
@[simps!]
def rightFunc : Arrow C ⥤ C :=
  Comma.snd _ _


/-- The natural transformation from `leftFunc` to `rightFunc`, given by the arrow itself. -/
@[simps]
def leftToRight : (leftFunc : Arrow C ⥤ C) ⟶ rightFunc where app f := f.hom


/-- A functor `C ⥤ D` induces a functor between the corresponding arrow categories. -/
@[simps]
def mapArrow (F : C ⥤ D) : Arrow C ⥤ Arrow D where
  obj a :=
    { left := F.obj a.left
      right := F.obj a.right
      hom := F.map a.hom }
  map f :=
    { left := F.map f.left
      right := F.map f.right
      w := by
        /-
          T : Type u
          inst✝² : CategoryTheory.Category.{v, u} T
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          X✝ Y✝ : CategoryTheory.Arrow C
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id D).map (F …
        -/
        let w := f.w
        /-
          T : Type u
          inst✝² : CategoryTheory.Category.{v, u} T
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          X✝ Y✝ : CategoryTheory.Arrow C
          f : Quiver.Hom X✝ Y✝
          w : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id C).map  …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id D).map (F …
        -/
        simp only [id_map] at w
        /-
          T : Type u
          inst✝² : CategoryTheory.Category.{v, u} T
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          X✝ Y✝ : CategoryTheory.Arrow C
          f : Quiver.Hom X✝ Y✝
          w : Eq (CategoryTheory.CategoryStruct.comp f.left Y✝.hom) (CategoryTheory.Cate …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id D).map (F …
        -/
        dsimp
        /-
          T : Type u
          inst✝² : CategoryTheory.Category.{v, u} T
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          X✝ Y✝ : CategoryTheory.Arrow C
          f : Quiver.Hom X✝ Y✝
          w : Eq (CategoryTheory.CategoryStruct.comp f.left Y✝.hom) (CategoryTheory.Cate …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f.left) (F.map Y✝.hom)) (Categ …
        -/
        simp only [← F.map_comp, w] }
        /-
          🎉 no goals
        -/


/-- The functor `(C ⥤ D) ⥤ (Arrow C ⥤ Arrow D)` which sends
a functor `F : C ⥤ D` to `F.mapArrow`. -/
@[simps]
def mapArrowFunctor : (C ⥤ D) ⥤ (Arrow C ⥤ Arrow D) where
  obj F := F.mapArrow
  map τ :=
    { app := fun f =>
        { left := τ.app _
          right := τ.app _ } }


/-- The equivalence of categories `Arrow C ≌ Arrow D` induced by an equivalence `C ≌ D`. -/
def mapArrowEquivalence (e : C ≌ D) : Arrow C ≌ Arrow D where
  functor := e.functor.mapArrow
  inverse := e.inverse.mapArrow
  unitIso := Functor.mapIso (mapArrowFunctor C C) e.unitIso
  counitIso := Functor.mapIso (mapArrowFunctor D D) e.counitIso


instance isEquivalence_mapArrow (F : C ⥤ D) [IsEquivalence F] :
    IsEquivalence F.mapArrow :=
  (mapArrowEquivalence (asEquivalence F)).isEquivalence_functor


/-- The images of `f : Arrow C` by two isomorphic functors `F : C ⥤ D` are
isomorphic arrows in `D`. -/
def Arrow.isoOfNatIso {C D : Type*} [Category C] [Category D] {F G : C ⥤ D} (e : F ≅ G)
    (f : Arrow C) : F.mapArrow.obj f ≅ G.mapArrow.obj f :=
  /-
    T : Type u
    inst✝² : CategoryTheory.Category.{v, u} T
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{?u.52381, u_1} C
    inst✝ : CategoryTheory.Category.{?u.52385, u_2} D
    F G : CategoryTheory.Functor C D
    e : CategoryTheory.Iso F G
    f : CategoryTheory.Arrow C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.app f.left).hom (G.mapArrow.obj f) …
  -/
  Arrow.isoMk (e.app f.left) (e.app f.right)
  /-
    🎉 no goals
  -/


