/-- A set of arrows all with codomain `X`. -/
def Presieve (X : C) :=
  ∀ ⦃Y⦄, Set (Y ⟶ X)-- deriving CompleteLattice


instance : CompleteLattice (Presieve X) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y Z : C
    f : Quiver.Hom Y X
    ⊢ CompleteLattice (CategoryTheory.Presieve X)
  -/
  dsimp [Presieve]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y Z : C
    f : Quiver.Hom Y X
    ⊢ CompleteLattice (⦃Y : C⦄ → Set (Quiver.Hom Y X))
  -/
  infer_instance
  /-
    🎉 no goals
  -/


noncomputable instance : Inhabited (Presieve X) :=
  ⟨⊤⟩


/-- The full subcategory of the over category `C/X` consisting of arrows which belong to a
    presieve on `X`. -/
abbrev category {X : C} (P : Presieve X) :=
  FullSubcategory fun f : Over X => P f.hom


/-- Construct an object of `P.category`. -/
abbrev categoryMk {X : C} (P : Presieve X) {Y : C} (f : Y ⟶ X) (hf : P f) : P.category :=
  ⟨Over.mk f, hf⟩


/-- Given a sieve `S` on `X : C`, its associated diagram `S.diagram` is defined to be
    the natural functor from the full subcategory of the over category `C/X` consisting
    of arrows in `S` to `C`. -/
abbrev diagram (S : Presieve X) : S.category ⥤ C :=
  fullSubcategoryInclusion _ ⋙ Over.forget X


/-- Given a sieve `S` on `X : C`, its associated cocone `S.cocone` is defined to be
    the natural cocone over the diagram defined above with cocone point `X`. -/
abbrev cocone (S : Presieve X) : Cocone S.diagram :=
  (Over.forgetCocone X).whisker (fullSubcategoryInclusion _)


/-- Given a set of arrows `S` all with codomain `X`, and a set of arrows with codomain `Y` for each
`f : Y ⟶ X` in `S`, produce a set of arrows with codomain `X`:
`{ g ≫ f | (f : Y ⟶ X) ∈ S, (g : Z ⟶ Y) ∈ R f }`.
-/
def bind (S : Presieve X) (R : ∀ ⦃Y⦄ ⦃f : Y ⟶ X⦄, S f → Presieve Y) : Presieve X := fun Z h =>
  ∃ (Y : C) (g : Z ⟶ Y) (f : Y ⟶ X) (H : S f), R H g ∧ g ≫ f = h


/-- Structure which contains the data and properties for a morphism `h` satisfying
`Presieve.bind S R h`. -/
structure BindStruct (S : Presieve X) (R : ∀ ⦃Y⦄ ⦃f : Y ⟶ X⦄, S f → Presieve Y)
    {Z : C} (h : Z ⟶ X) where
  /-- the intermediate object -/
  Y : C
  /-- a morphism in the family of presieves `R` -/
  g : Z ⟶ Y
  /-- a morphism in the presieve `S` -/
  f : Y ⟶ X
  hf : S f
  hg : R hf g
  fac : g ≫ f = h


attribute [reassoc (attr := simp)] BindStruct.fac


/-- If a morphism `h` satisfies `Presieve.bind S R h`, this is a choice of a structure
in `BindStruct S R h`. -/
noncomputable def bind.bindStruct {S : Presieve X} {R : ∀ ⦃Y⦄ ⦃f : Y ⟶ X⦄, S f → Presieve Y}
    {Z : C} {h : Z ⟶ X} (H : bind S R h) : BindStruct S R h :=
  Nonempty.some (by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y Z✝ : C
      f : Quiver.Hom Y X
      S : CategoryTheory.Presieve X
      R : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → S f → CategoryTheory.Presieve Y
      Z : C
      h : Quiver.Hom Z X
      H : S.bind R h
      ⊢ Nonempty (S.BindStruct R h)
    -/
    obtain ⟨Y, g, f, hf, hg, fac⟩ := H
    /-
      case intro.intro.intro.intro.intro
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y✝ Z✝ : C
      f✝ : Quiver.Hom Y✝ X
      S : CategoryTheory.Presieve X
      R : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → S f → CategoryTheory.Presieve Y
      Z : C
      h : Quiver.Hom Z X
      Y : C
      g : Quiver.Hom Z Y
      f : Quiver.Hom Y X
      hf : S f
      hg : R hf g
      fac : Eq (CategoryTheory.CategoryStruct.comp g f) h
      ⊢ Nonempty (S.BindStruct R h)
    -/
    exact ⟨{ hf := hf, hg := hg, fac := fac }⟩)
    /-
      🎉 no goals
    -/


lemma BindStruct.bind {S : Presieve X} {R : ∀ ⦃Y⦄ ⦃f : Y ⟶ X⦄, S f → Presieve Y}
    {Z : C} {h : Z ⟶ X} (b : BindStruct S R h) : bind S R h :=
  ⟨b.Y, b.g, b.f, b.hf, b.hg, b.fac⟩


@[simp]
theorem bind_comp {S : Presieve X} {R : ∀ ⦃Y : C⦄ ⦃f : Y ⟶ X⦄, S f → Presieve Y} {g : Z ⟶ Y}
    (h₁ : S f) (h₂ : R h₁ g) : bind S R (g ≫ f) :=
  ⟨_, _, _, h₁, h₂, rfl⟩

-- Porting note: it seems the definition of `Presieve` must be unfolded in order to define
--   this inductive type, it was thus renamed `singleton'`
-- Note we can't make this into `HasSingleton` because of the out-param.

/-- The singleton presieve. -/
inductive singleton' : ⦃Y : C⦄ → (Y ⟶ X) → Prop
  | mk : singleton' f


/-- The singleton presieve. -/
def singleton : Presieve X := singleton' f


lemma singleton.mk {f : Y ⟶ X} : singleton f f := singleton'.mk


@[simp]
theorem singleton_eq_iff_domain (f g : Y ⟶ X) : singleton f g ↔ f = g := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f g : Quiver.Hom Y X
    ⊢ Iff (CategoryTheory.Presieve.singleton f g) (Eq f g)
  -/
  constructor
    /-
      case mp
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      f g : Quiver.Hom Y X
      ⊢ CategoryTheory.Presieve.singleton f g → Eq f g
    -/
  · rintro ⟨a, rfl⟩
    /-
      case mp.mk
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y✝ : C
      f : Quiver.Hom Y✝ X
      Y : C
      ⊢ Eq f f
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      f g : Quiver.Hom Y X
      ⊢ Eq f g → CategoryTheory.Presieve.singleton f g
    -/
  · rintro rfl
    /-
      case mpr
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      f : Quiver.Hom Y X
      ⊢ CategoryTheory.Presieve.singleton f f
    -/
    apply singleton.mk
    /-
      🎉 no goals
    -/


theorem singleton_self : singleton f f :=
  singleton.mk


/-- Pullback a set of arrows with given codomain along a fixed map, by taking the pullback in the
category.
This is not the same as the arrow set of `Sieve.pullback`, but there is a relation between them
in `pullbackArrows_comm`.
-/
inductive pullbackArrows [HasPullbacks C] (R : Presieve X) : Presieve Y
  | mk (Z : C) (h : Z ⟶ X) : R h → pullbackArrows _ (pullback.snd h f)


theorem pullback_singleton [HasPullbacks C] (g : Z ⟶ X) :
    pullbackArrows f (singleton g) = singleton (pullback.snd g f) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    f : Quiver.Hom Y X
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    g : Quiver.Hom Z X
    ⊢ Eq (CategoryTheory.Presieve.pullbackArrows f (CategoryTheory.Presieve.single …
  -/
  funext W
  /-
    case h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    f : Quiver.Hom Y X
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    g : Quiver.Hom Z X
    W : C
    ⊢ Eq (CategoryTheory.Presieve.pullbackArrows f (CategoryTheory.Presieve.single …
  -/
  ext h
  /-
    case h.h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    f : Quiver.Hom Y X
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    g : Quiver.Hom Z X
    W : C
    h : Quiver.Hom W Y
    ⊢ Iff (Membership.mem (CategoryTheory.Presieve.pullbackArrows f (CategoryTheor …
  -/
  constructor
    /-
      case h.h.mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      f : Quiver.Hom Y X
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      g : Quiver.Hom Z X
      W : C
      h : Quiver.Hom W Y
      ⊢ Membership.mem (CategoryTheory.Presieve.pullbackArrows f (CategoryTheory.Pre …
    -/
  · rintro ⟨W, _, _, _⟩
    /-
      case h.h.mp.mk.mk
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y✝¹ Z : C
      f : Quiver.Hom Y✝¹ X
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      g : Quiver.Hom Z X
      Y✝ Y : C
      ⊢ Membership.mem (CategoryTheory.Presieve.singleton (CategoryTheory.Limits.pul …
    -/
    exact singleton.mk
    /-
      🎉 no goals
    -/
    /-
      case h.h.mpr
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      f : Quiver.Hom Y X
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      g : Quiver.Hom Z X
      W : C
      h : Quiver.Hom W Y
      ⊢ Membership.mem (CategoryTheory.Presieve.singleton (CategoryTheory.Limits.pul …
    -/
  · rintro ⟨_⟩
    /-
      case h.h.mpr.mk
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y✝ Z : C
      f : Quiver.Hom Y✝ X
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      g : Quiver.Hom Z X
      Y : C
      ⊢ Membership.mem (CategoryTheory.Presieve.pullbackArrows f (CategoryTheory.Pre …
    -/
    exact pullbackArrows.mk Z g singleton.mk
    /-
      🎉 no goals
    -/


/-- Construct the presieve given by the family of arrows indexed by `ι`. -/
inductive ofArrows {ι : Type*} (Y : ι → C) (f : ∀ i, Y i ⟶ X) : Presieve X
  | mk (i : ι) : ofArrows _ _ (f i)


theorem ofArrows_pUnit : (ofArrows _ fun _ : PUnit => f) = singleton f := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom Y X
    ⊢ Eq (CategoryTheory.Presieve.ofArrows (fun x => Y) fun x => f) (CategoryTheor …
  -/
  funext Y
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y✝ : C
    f : Quiver.Hom Y✝ X
    Y : C
    ⊢ Eq (CategoryTheory.Presieve.ofArrows (fun x => Y✝) fun x => f) (CategoryTheo …
  -/
  ext g
  /-
    case h.h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y✝ : C
    f : Quiver.Hom Y✝ X
    Y : C
    g : Quiver.Hom Y X
    ⊢ Iff (Membership.mem (CategoryTheory.Presieve.ofArrows (fun x => Y✝) fun x => …
  -/
  constructor
    /-
      case h.h.mp
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y✝ : C
      f : Quiver.Hom Y✝ X
      Y : C
      g : Quiver.Hom Y X
      ⊢ Membership.mem (CategoryTheory.Presieve.ofArrows (fun x => Y✝) fun x => f) g …
    -/
  · rintro ⟨_⟩
    /-
      case h.h.mp.mk
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y✝ : C
      f : Quiver.Hom Y✝ X
      Y : C
      i✝ : PUnit.{u_1 + 1}
      ⊢ Membership.mem (CategoryTheory.Presieve.singleton f) f
    -/
    apply singleton.mk
    /-
      🎉 no goals
    -/
    /-
      case h.h.mpr
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y✝ : C
      f : Quiver.Hom Y✝ X
      Y : C
      g : Quiver.Hom Y X
      ⊢ Membership.mem (CategoryTheory.Presieve.singleton f) g → Membership.mem (Cat …
    -/
  · rintro ⟨_⟩
    /-
      case h.h.mpr.mk
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y✝ : C
      f : Quiver.Hom Y✝ X
      Y : C
      ⊢ Membership.mem (CategoryTheory.Presieve.ofArrows (fun x => Y✝) fun x => f) f
    -/
    exact ofArrows.mk PUnit.unit
    /-
      🎉 no goals
    -/


theorem ofArrows_pullback [HasPullbacks C] {ι : Type*} (Z : ι → C) (g : ∀ i : ι, Z i ⟶ X) :
    (ofArrows (fun i => pullback (g i) f) fun _ => pullback.snd _ _) =
      pullbackArrows f (ofArrows Z g) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom Y X
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    ι : Type u_1
    Z : ι → C
    g : (i : ι) → Quiver.Hom (Z i) X
    ⊢ Eq (CategoryTheory.Presieve.ofArrows (fun i => CategoryTheory.Limits.pullbac …
  -/
  funext T
  /-
    case h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom Y X
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    ι : Type u_1
    Z : ι → C
    g : (i : ι) → Quiver.Hom (Z i) X
    T : C
    ⊢ Eq (CategoryTheory.Presieve.ofArrows (fun i => CategoryTheory.Limits.pullbac …
  -/
  ext h
  /-
    case h.h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom Y X
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    ι : Type u_1
    Z : ι → C
    g : (i : ι) → Quiver.Hom (Z i) X
    T : C
    h : Quiver.Hom T Y
    ⊢ Iff (Membership.mem (CategoryTheory.Presieve.ofArrows (fun i => CategoryTheo …
  -/
  constructor
    /-
      case h.h.mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      f : Quiver.Hom Y X
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      ι : Type u_1
      Z : ι → C
      g : (i : ι) → Quiver.Hom (Z i) X
      T : C
      h : Quiver.Hom T Y
      ⊢ Membership.mem (CategoryTheory.Presieve.ofArrows (fun i => CategoryTheory.Li …
    -/
  · rintro ⟨hk⟩
    /-
      case h.h.mp.mk
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y✝ : C
      f : Quiver.Hom Y✝ X
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      ι : Type u_1
      Z : ι → C
      g : (i : ι) → Quiver.Hom (Z i) X
      Y : C
      hk : ι
      ⊢ Membership.mem (CategoryTheory.Presieve.pullbackArrows f (CategoryTheory.Pre …
    -/
    exact pullbackArrows.mk _ _ (ofArrows.mk hk)
    /-
      🎉 no goals
    -/
    /-
      case h.h.mpr
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      f : Quiver.Hom Y X
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      ι : Type u_1
      Z : ι → C
      g : (i : ι) → Quiver.Hom (Z i) X
      T : C
      h : Quiver.Hom T Y
      ⊢ Membership.mem (CategoryTheory.Presieve.pullbackArrows f (CategoryTheory.Pre …
    -/
  · rintro ⟨W, k, hk₁⟩
    /-
      case h.h.mpr.mk
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y✝ : C
      f : Quiver.Hom Y✝ X
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      ι : Type u_1
      Z : ι → C
      g : (i : ι) → Quiver.Hom (Z i) X
      Y W : C
      k : Quiver.Hom W X
      hk₁ : CategoryTheory.Presieve.ofArrows Z g k
      ⊢ Membership.mem (CategoryTheory.Presieve.ofArrows (fun i => CategoryTheory.Li …
    -/
    cases' hk₁ with i hi
    /-
      case h.h.mpr.mk.mk
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y✝ : C
      f : Quiver.Hom Y✝ X
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      ι : Type u_1
      Z : ι → C
      g : (i : ι) → Quiver.Hom (Z i) X
      Y : C
      i : ι
      ⊢ Membership.mem (CategoryTheory.Presieve.ofArrows (fun i => CategoryTheory.Li …
    -/
    apply ofArrows.mk
    /-
      🎉 no goals
    -/


theorem ofArrows_bind {ι : Type*} (Z : ι → C) (g : ∀ i : ι, Z i ⟶ X)
    (j : ∀ ⦃Y⦄ (f : Y ⟶ X), ofArrows Z g f → Type*) (W : ∀ ⦃Y⦄ (f : Y ⟶ X) (H), j f H → C)
    (k : ∀ ⦃Y⦄ (f : Y ⟶ X) (H i), W f H i ⟶ Y) :
    ((ofArrows Z g).bind fun _ f H => ofArrows (W f H) (k f H)) =
      ofArrows (fun i : Σi, j _ (ofArrows.mk i) => W (g i.1) _ i.2) fun ij =>
        k (g ij.1) _ ij.2 ≫ g ij.1 := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    ι : Type u_1
    Z : ι → C
    g : (i : ι) → Quiver.Hom (Z i) X
    j : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Presieve.ofArrows Z g f →  …
    W : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → (H : CategoryTheory.Presieve.ofArrows Z g …
    k : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → (H : CategoryTheory.Presieve.ofArrows Z g …
    ⊢ Eq ((CategoryTheory.Presieve.ofArrows Z g).bind fun x f H => CategoryTheory. …
  -/
  funext Y
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    ι : Type u_1
    Z : ι → C
    g : (i : ι) → Quiver.Hom (Z i) X
    j : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Presieve.ofArrows Z g f →  …
    W : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → (H : CategoryTheory.Presieve.ofArrows Z g …
    k : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → (H : CategoryTheory.Presieve.ofArrows Z g …
    Y : C
    ⊢ Eq ((CategoryTheory.Presieve.ofArrows Z g).bind fun x f H => CategoryTheory. …
  -/
  ext f
  /-
    case h.h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    ι : Type u_1
    Z : ι → C
    g : (i : ι) → Quiver.Hom (Z i) X
    j : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Presieve.ofArrows Z g f →  …
    W : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → (H : CategoryTheory.Presieve.ofArrows Z g …
    k : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → (H : CategoryTheory.Presieve.ofArrows Z g …
    Y : C
    f : Quiver.Hom Y X
    ⊢ Iff (Membership.mem ((CategoryTheory.Presieve.ofArrows Z g).bind fun x f H = …
  -/
  constructor
    /-
      case h.h.mp
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      ι : Type u_1
      Z : ι → C
      g : (i : ι) → Quiver.Hom (Z i) X
      j : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Presieve.ofArrows Z g f →  …
      W : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → (H : CategoryTheory.Presieve.ofArrows Z g …
      k : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → (H : CategoryTheory.Presieve.ofArrows Z g …
      Y : C
      f : Quiver.Hom Y X
      ⊢ Membership.mem ((CategoryTheory.Presieve.ofArrows Z g).bind fun x f H => Cat …
    -/
  · rintro ⟨_, _, _, ⟨i⟩, ⟨i'⟩, rfl⟩
    /-
      case h.h.mp.intro.intro.intro.intro.mk.intro.mk
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      ι : Type u_1
      Z : ι → C
      g : (i : ι) → Quiver.Hom (Z i) X
      j : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Presieve.ofArrows Z g f →  …
      W : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → (H : CategoryTheory.Presieve.ofArrows Z g …
      k : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → (H : CategoryTheory.Presieve.ofArrows Z g …
      Y✝ : C
      i : ι
      Y : C
      i' : j (g i) ⋯
      ⊢ Membership.mem (CategoryTheory.Presieve.ofArrows (fun i => W (g i.fst) ⋯ i.s …
    -/
    exact ofArrows.mk (Sigma.mk _ _)
    /-
      🎉 no goals
    -/
    /-
      case h.h.mpr
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      ι : Type u_1
      Z : ι → C
      g : (i : ι) → Quiver.Hom (Z i) X
      j : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Presieve.ofArrows Z g f →  …
      W : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → (H : CategoryTheory.Presieve.ofArrows Z g …
      k : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → (H : CategoryTheory.Presieve.ofArrows Z g …
      Y : C
      f : Quiver.Hom Y X
      ⊢ Membership.mem (CategoryTheory.Presieve.ofArrows (fun i => W (g i.fst) ⋯ i.s …
    -/
  · rintro ⟨i⟩
    /-
      case h.h.mpr.mk
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      ι : Type u_1
      Z : ι → C
      g : (i : ι) → Quiver.Hom (Z i) X
      j : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Presieve.ofArrows Z g f →  …
      W : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → (H : CategoryTheory.Presieve.ofArrows Z g …
      k : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → (H : CategoryTheory.Presieve.ofArrows Z g …
      Y : C
      i : Sigma fun i => j (g i) ⋯
      ⊢ Membership.mem ((CategoryTheory.Presieve.ofArrows Z g).bind fun x f H => Cat …
    -/
    exact bind_comp _ (ofArrows.mk _) (ofArrows.mk _)
    /-
      🎉 no goals
    -/


theorem ofArrows_surj {ι : Type*} {Y : ι → C} (f : ∀ i, Y i ⟶ X) {Z : C} (g : Z ⟶ X)
    (hg : ofArrows Y f g) : ∃ (i : ι) (h : Y i = Z),
    g = eqToHom h.symm ≫ f i := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    ι : Type u_1
    Y : ι → C
    f : (i : ι) → Quiver.Hom (Y i) X
    Z : C
    g : Quiver.Hom Z X
    hg : CategoryTheory.Presieve.ofArrows Y f g
    ⊢ Exists fun i => Exists fun h => Eq g (CategoryTheory.CategoryStruct.comp (Ca …
  -/
  cases' hg with i
  /-
    case mk
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    ι : Type u_1
    Y : ι → C
    f : (i : ι) → Quiver.Hom (Y i) X
    i : ι
    ⊢ Exists fun i_1 => Exists fun h => Eq (f i) (CategoryTheory.CategoryStruct.co …
  -/
  exact ⟨i, rfl, by simp only [eqToHom_refl, id_comp]⟩
  /-
    🎉 no goals
  -/


/-- Given a presieve on `F(X)`, we can define a presieve on `X` by taking the preimage via `F`. -/
def functorPullback (R : Presieve (F.obj X)) : Presieve X := fun _ f => R (F.map f)


@[simp]
theorem functorPullback_mem (R : Presieve (F.obj X)) {Y} (f : Y ⟶ X) :
    R.functorPullback F f ↔ R (F.map f) :=
  Iff.rfl


@[simp]
theorem functorPullback_id (R : Presieve X) : R.functorPullback (𝟭 _) = R :=
  rfl


/-- Given a presieve `R` on `X`, the predicate `R.hasPullbacks` means that for all arrows `f` and
    `g` in `R`, the pullback of `f` and `g` exists. -/
class hasPullbacks (R : Presieve X) : Prop where
  /-- For all arrows `f` and `g` in `R`, the pullback of `f` and `g` exists. -/
  has_pullbacks : ∀ {Y Z} {f : Y ⟶ X} (_ : R f) {g : Z ⟶ X} (_ : R g), HasPullback f g


instance (R : Presieve X) [HasPullbacks C] : R.hasPullbacks := ⟨fun _ _ ↦ inferInstance⟩


instance {α : Type v₂} {X : α → C} {B : C} (π : (a : α) → X a ⟶ B)
    [(Presieve.ofArrows X π).hasPullbacks] (a b : α) : HasPullback (π a) (π b) :=
  Presieve.hasPullbacks.has_pullbacks (Presieve.ofArrows.mk _) (Presieve.ofArrows.mk _)


/-- Given a presieve on `X`, we can define a presieve on `F(X)` (which is actually a sieve)
by taking the sieve generated by the image via `F`.
-/
def functorPushforward (S : Presieve X) : Presieve (F.obj X) := fun Y f =>
  ∃ (Z : C) (g : Z ⟶ X) (h : Y ⟶ F.obj Z), S g ∧ f = h ≫ F.map g

-- Porting note: removed @[nolint hasNonemptyInstance]

/-- An auxiliary definition in order to fix the choice of the preimages between various definitions.
-/
structure FunctorPushforwardStructure (S : Presieve X) {Y} (f : Y ⟶ F.obj X) where
  /-- an object in the source category -/
  preobj : C
  /-- a map in the source category which has to be in the presieve -/
  premap : preobj ⟶ X
  /-- the morphism which appear in the factorisation -/
  lift : Y ⟶ F.obj preobj
  /-- the condition that `premap` is in the presieve -/
  cover : S premap
  /-- the factorisation of the morphism -/
  fac : f = lift ≫ F.map premap


/-- The fixed choice of a preimage. -/
noncomputable def getFunctorPushforwardStructure {F : C ⥤ D} {S : Presieve X} {Y : D}
    {f : Y ⟶ F.obj X} (h : S.functorPushforward F f) : FunctorPushforwardStructure F S f := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F✝ : CategoryTheory.Functor C D
    X Y✝ Z : C
    f✝ : Quiver.Hom Y✝ X
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    F : CategoryTheory.Functor C D
    S : CategoryTheory.Presieve X
    Y : D
    f : Quiver.Hom Y (F.obj X)
    h : CategoryTheory.Presieve.functorPushforward F S f
    ⊢ CategoryTheory.Presieve.FunctorPushforwardStructure F S f
  -/
  choose Z f' g h₁ h using h
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F✝ : CategoryTheory.Functor C D
    X Y✝ Z✝ : C
    f✝ : Quiver.Hom Y✝ X
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    F : CategoryTheory.Functor C D
    S : CategoryTheory.Presieve X
    Y : D
    f : Quiver.Hom Y (F.obj X)
    Z : C
    f' : Quiver.Hom Z X
    g : Quiver.Hom Y (F.obj Z)
    h₁ : S f'
    h : Eq f (CategoryTheory.CategoryStruct.comp g (F.map f'))
    ⊢ CategoryTheory.Presieve.FunctorPushforwardStructure F S f
  -/
  exact ⟨Z, f', g, h₁, h⟩
  /-
    🎉 no goals
  -/


theorem functorPushforward_comp (R : Presieve X) :
    R.functorPushforward (F ⋙ G) = (R.functorPushforward F).functorPushforward G := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X : C
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    R : CategoryTheory.Presieve X
    ⊢ Eq (CategoryTheory.Presieve.functorPushforward (F.comp G) R) (CategoryTheory …
  -/
  funext x
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X : C
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    R : CategoryTheory.Presieve X
    x : E
    ⊢ Eq (CategoryTheory.Presieve.functorPushforward (F.comp G) R) (CategoryTheory …
  -/
  ext f
  /-
    case h.h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X : C
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    R : CategoryTheory.Presieve X
    x : E
    f : Quiver.Hom x ((F.comp G).obj X)
    ⊢ Iff (Membership.mem (CategoryTheory.Presieve.functorPushforward (F.comp G) R …
  -/
  constructor
    /-
      case h.h.mp
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X : C
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      R : CategoryTheory.Presieve X
      x : E
      f : Quiver.Hom x ((F.comp G).obj X)
      ⊢ Membership.mem (CategoryTheory.Presieve.functorPushforward (F.comp G) R) f → …
    -/
  · rintro ⟨X, f₁, g₁, h₁, rfl⟩
    /-
      case h.h.mp.intro.intro.intro.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X✝ : C
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      R : CategoryTheory.Presieve X✝
      x : E
      X : C
      f₁ : Quiver.Hom X X✝
      g₁ : Quiver.Hom x ((F.comp G).obj X)
      h₁ : R f₁
      ⊢ Membership.mem (CategoryTheory.Presieve.functorPushforward G (CategoryTheory …
    -/
    exact ⟨F.obj X, F.map f₁, g₁, ⟨X, f₁, 𝟙 _, h₁, by simp⟩, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case h.h.mpr
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X : C
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      R : CategoryTheory.Presieve X
      x : E
      f : Quiver.Hom x ((F.comp G).obj X)
      ⊢ Membership.mem (CategoryTheory.Presieve.functorPushforward G (CategoryTheory …
    -/
  · rintro ⟨X, f₁, g₁, ⟨X', f₂, g₂, h₁, rfl⟩, rfl⟩
    /-
      case h.h.mpr.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X✝ : C
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      R : CategoryTheory.Presieve X✝
      x : E
      X : D
      g₁ : Quiver.Hom x (G.obj X)
      X' : C
      f₂ : Quiver.Hom X' X✝
      g₂ : Quiver.Hom X (F.obj X')
      h₁ : R f₂
      ⊢ Membership.mem (CategoryTheory.Presieve.functorPushforward (F.comp G) R) (Ca …
    -/
    exact ⟨X', f₂, g₁ ≫ G.map g₂, h₁, by simp⟩
    /-
      🎉 no goals
    -/


theorem image_mem_functorPushforward (R : Presieve X) {f : Y ⟶ X} (h : R f) :
    R.functorPushforward F (F.map f) :=
                    /-
                      C : Type u₁
                      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                      D : Type u₂
                      inst✝ : CategoryTheory.Category.{v₂, u₂} D
                      F : CategoryTheory.Functor C D
                      X Y : C
                      R : CategoryTheory.Presieve X
                      f : Quiver.Hom Y X
                      h : R f
                      ⊢ Eq (F.map f) (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStr …
                    -/
  ⟨Y, f, 𝟙 _, h, by simp⟩
                    /-
                      🎉 no goals
                    -/


/--
For an object `X` of a category `C`, a `Sieve X` is a set of morphisms to `X` which is closed under
left-composition.
-/
structure Sieve {C : Type u₁} [Category.{v₁} C] (X : C) where
  /-- the underlying presieve -/
  arrows : Presieve X
  /-- stability by precomposition -/
  downward_closed : ∀ {Y Z f} (_ : arrows f) (g : Z ⟶ Y), arrows (g ≫ f)


instance : CoeFun (Sieve X) fun _ => Presieve X :=
  ⟨Sieve.arrows⟩


attribute [simp] downward_closed


theorem arrows_ext : ∀ {R S : Sieve X}, R.arrows = S.arrows → R = S := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    ⊢ ∀ {R S : CategoryTheory.Sieve X}, Eq R.arrows S.arrows → Eq R S
  -/
  rintro ⟨_, _⟩ ⟨_, _⟩ rfl
  /-
    case mk.mk
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    arrows✝ : CategoryTheory.Presieve X
    downward_closed✝¹ : ∀ {Y Z : C} {f : Quiver.Hom Y X}, arrows✝ f → ∀ (g : Quive …
    downward_closed✝ : ∀ {Y Z : C} {f : Quiver.Hom Y X}, { arrows := arrows✝, down …
    ⊢ Eq { arrows := arrows✝, downward_closed := downward_closed✝¹ } { arrows := { …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[ext]
protected theorem ext {R S : Sieve X} (h : ∀ ⦃Y⦄ (f : Y ⟶ X), R f ↔ S f) : R = S :=
  arrows_ext <| funext fun _ => funext fun f => propext <| h f


/-- The supremum of a collection of sieves: the union of them all. -/
protected def sup (𝒮 : Set (Sieve X)) : Sieve X where
  arrows _ := { f | ∃ S ∈ 𝒮, Sieve.arrows S f }
  downward_closed {_ _ f} hf _ := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y Z : C
      f✝ : Quiver.Hom Y X
      S R : CategoryTheory.Sieve X
      𝒮 : Set (CategoryTheory.Sieve X)
      x✝² x✝¹ : C
      f : Quiver.Hom x✝² X
      hf : (fun x => setOf fun f => Exists fun S => And (Membership.mem 𝒮 S) (S.arro …
      x✝ : Quiver.Hom x✝¹ x✝²
      ⊢ (fun x => setOf fun f => Exists fun S => And (Membership.mem 𝒮 S) (S.arrows  …
    -/
    obtain ⟨S, hS, hf⟩ := hf
    /-
      case intro.intro
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y Z : C
      f✝ : Quiver.Hom Y X
      S✝ R : CategoryTheory.Sieve X
      𝒮 : Set (CategoryTheory.Sieve X)
      x✝² x✝¹ : C
      f : Quiver.Hom x✝² X
      x✝ : Quiver.Hom x✝¹ x✝²
      S : CategoryTheory.Sieve X
      hS : Membership.mem 𝒮 S
      hf : S.arrows f
      ⊢ setOf (fun f => Exists fun S => And (Membership.mem 𝒮 S) (S.arrows f)) (Cate …
    -/
    exact ⟨S, hS, S.downward_closed hf _⟩
    /-
      🎉 no goals
    -/


/-- The infimum of a collection of sieves: the intersection of them all. -/
protected def inf (𝒮 : Set (Sieve X)) : Sieve X where
  arrows _ := { f | ∀ S ∈ 𝒮, Sieve.arrows S f }
  downward_closed {_ _ _} hf g S H := S.downward_closed (hf S H) g


/-- The union of two sieves is a sieve. -/
protected def union (S R : Sieve X) : Sieve X where
  arrows _ f := S f ∨ R f
                        /-
                          C : Type u₁
                          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                          D : Type u₂
                          inst✝ : CategoryTheory.Category.{v₂, u₂} D
                          F : CategoryTheory.Functor C D
                          X Y Z : C
                          f : Quiver.Hom Y X
                          S✝ R✝ S R : CategoryTheory.Sieve X
                          ⊢ ∀ {Y Z : C} {f : Quiver.Hom Y X}, (fun x f => Or (S.arrows f) (R.arrows f))  …
                        -/
                                                   /-
                                                     🎉 no goals
                                                   -/
  downward_closed := by rintro _ _ _ (h | h) g <;> simp [h]
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- The intersection of two sieves is a sieve. -/
protected def inter (S R : Sieve X) : Sieve X where
  arrows _ f := S f ∧ R f
  downward_closed := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y Z : C
      f : Quiver.Hom Y X
      S✝ R✝ S R : CategoryTheory.Sieve X
      ⊢ ∀ {Y Z : C} {f : Quiver.Hom Y X}, (fun x f => And (S.arrows f) (R.arrows f)) …
    -/
    rintro _ _ _ ⟨h₁, h₂⟩ g
    /-
      case intro
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y Z : C
      f : Quiver.Hom Y X
      S✝ R✝ S R : CategoryTheory.Sieve X
      Y✝ Z✝ : C
      f✝ : Quiver.Hom Y✝ X
      h₁ : S.arrows f✝
      h₂ : R.arrows f✝
      g : Quiver.Hom Z✝ Y✝
      ⊢ And (S.arrows (CategoryTheory.CategoryStruct.comp g f✝)) (R.arrows (Category …
    -/
    simp [h₁, h₂]
    /-
      🎉 no goals
    -/


/-- Sieves on an object `X` form a complete lattice.
We generate this directly rather than using the galois insertion for nicer definitional properties.
-/
instance : CompleteLattice (Sieve X) where
  le S R := ∀ ⦃Y⦄ (f : Y ⟶ X), S f → R f
  le_refl _ _ _ := id
  le_trans _ _ _ S₁₂ S₂₃ _ _ h := S₂₃ _ (S₁₂ _ h)
  le_antisymm _ _ p q := Sieve.ext fun _ _ => ⟨p _, q _⟩
  top :=
    { arrows := fun _ => Set.univ
      downward_closed := fun _ _ => ⟨⟩ }
  bot :=
    { arrows := fun _ => ∅
      downward_closed := False.elim }
  sup := Sieve.union
  inf := Sieve.inter
  sSup := Sieve.sup
  sInf := Sieve.inf
  le_sSup _ S hS _ _ hf := ⟨S, hS, hf⟩
  sSup_le := fun _ _ ha _ _ ⟨b, hb, hf⟩ => (ha b hb) _ hf
  sInf_le _ _ hS _ _ h := h _ hS
  le_sInf _ _ hS _ _ hf _ hR := hS _ hR _ hf
  le_sup_left _ _ _ _ := Or.inl
  le_sup_right _ _ _ _ := Or.inr
  sup_le _ _ _ h₁ h₂ _ f := by--ℰ S hS Y f := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y Z : C
      f✝ : Quiver.Hom Y X
      S R x✝³ x✝² x✝¹ : CategoryTheory.Sieve X
      h₁ : LE.le x✝³ x✝¹
      h₂ : LE.le x✝² x✝¹
      x✝ : C
      f : Quiver.Hom x✝ X
      ⊢ (x✝³.union x✝²).arrows f → x✝¹.arrows f
    -/
    rintro (hf | hf)
      /-
        case inl
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        X Y Z : C
        f✝ : Quiver.Hom Y X
        S R x✝³ x✝² x✝¹ : CategoryTheory.Sieve X
        h₁ : LE.le x✝³ x✝¹
        h₂ : LE.le x✝² x✝¹
        x✝ : C
        f : Quiver.Hom x✝ X
        hf : x✝³.arrows f
        ⊢ x✝¹.arrows f
      -/
    · exact h₁ _ hf
      /-
        🎉 no goals
      -/
      /-
        case inr
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        X Y Z : C
        f✝ : Quiver.Hom Y X
        S R x✝³ x✝² x✝¹ : CategoryTheory.Sieve X
        h₁ : LE.le x✝³ x✝¹
        h₂ : LE.le x✝² x✝¹
        x✝ : C
        f : Quiver.Hom x✝ X
        hf : x✝².arrows f
        ⊢ x✝¹.arrows f
      -/
    · exact h₂ _ hf
      /-
        🎉 no goals
      -/
  inf_le_left _ _ _ _ := And.left
  inf_le_right _ _ _ _ := And.right
  le_inf _ _ _ p q _ _ z := ⟨p _ z, q _ z⟩
  le_top _ _ _ _ := trivial
  bot_le _ _ _ := False.elim


/-- The maximal sieve always exists. -/
instance sieveInhabited : Inhabited (Sieve X) :=
  ⟨⊤⟩


@[simp]
theorem sInf_apply {Ss : Set (Sieve X)} {Y} (f : Y ⟶ X) :
    sInf Ss f ↔ ∀ (S : Sieve X) (_ : S ∈ Ss), S f :=
  Iff.rfl


@[simp]
theorem sSup_apply {Ss : Set (Sieve X)} {Y} (f : Y ⟶ X) :
    sSup Ss f ↔ ∃ (S : Sieve X) (_ : S ∈ Ss), S f := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    Ss : Set (CategoryTheory.Sieve X)
    Y : C
    f : Quiver.Hom Y X
    ⊢ Iff ((SupSet.sSup Ss).arrows f) (Exists fun S => Exists fun x => S.arrows f)
  -/
  simp [sSup, Sieve.sup, setOf]
  /-
    🎉 no goals
  -/


@[simp]
theorem inter_apply {R S : Sieve X} {Y} (f : Y ⟶ X) : (R ⊓ S) f ↔ R f ∧ S f :=
  Iff.rfl


@[simp]
theorem union_apply {R S : Sieve X} {Y} (f : Y ⟶ X) : (R ⊔ S) f ↔ R f ∨ S f :=
  Iff.rfl


@[simp]
theorem top_apply (f : Y ⟶ X) : (⊤ : Sieve X) f :=
  trivial


/-- Generate the smallest sieve containing the given set of arrows. -/
@[simps]
def generate (R : Presieve X) : Sieve X where
  arrows Z f := ∃ (Y : _) (h : Z ⟶ Y) (g : Y ⟶ X), R g ∧ h ≫ g = f
  downward_closed := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y Z : C
      f : Quiver.Hom Y X
      S R✝ : CategoryTheory.Sieve X
      R : CategoryTheory.Presieve X
      ⊢ ∀ {Y Z : C} {f : Quiver.Hom Y X}, (fun Z f => Exists fun Y => Exists fun h = …
    -/
    rintro Y Z _ ⟨W, g, f, hf, rfl⟩ h
    /-
      case intro.intro.intro.intro
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y✝ Z✝ : C
      f✝ : Quiver.Hom Y✝ X
      S R✝ : CategoryTheory.Sieve X
      R : CategoryTheory.Presieve X
      Y Z W : C
      g : Quiver.Hom Y W
      f : Quiver.Hom W X
      hf : R f
      h : Quiver.Hom Z Y
      ⊢ Exists fun Y_1 => Exists fun h_1 => Exists fun g_1 => And (R g_1) (Eq (Categ …
    -/
    exact ⟨_, h ≫ g, _, hf, by simp⟩
    /-
      🎉 no goals
    -/


/-- Given a presieve on `X`, and a sieve on each domain of an arrow in the presieve, we can bind to
produce a sieve on `X`.
-/
@[simps]
def bind (S : Presieve X) (R : ∀ ⦃Y⦄ ⦃f : Y ⟶ X⦄, S f → Sieve Y) : Sieve X where
  arrows := S.bind fun _ _ h => R h
  downward_closed := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y Z : C
      f : Quiver.Hom Y X
      S✝ R✝ : CategoryTheory.Sieve X
      S : CategoryTheory.Presieve X
      R : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → S f → CategoryTheory.Sieve Y
      ⊢ ∀ {Y Z : C} {f : Quiver.Hom Y X}, S.bind (fun x x_1 h => (R h).arrows) f → ∀ …
    -/
    rintro Y Z f ⟨W, f, h, hh, hf, rfl⟩ g
    /-
      case intro.intro.intro.intro.intro
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y✝ Z✝ : C
      f✝ : Quiver.Hom Y✝ X
      S✝ R✝ : CategoryTheory.Sieve X
      S : CategoryTheory.Presieve X
      R : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → S f → CategoryTheory.Sieve Y
      Y Z W : C
      f : Quiver.Hom Y W
      h : Quiver.Hom W X
      hh : S h
      hf : (R hh).arrows f
      g : Quiver.Hom Z Y
      ⊢ S.bind (fun x x_1 h => (R h).arrows) (CategoryTheory.CategoryStruct.comp g ( …
    -/
    exact ⟨_, g ≫ f, _, hh, by simp [hf]⟩
    /-
      🎉 no goals
    -/


/-- Structure which contains the data and properties for a morphism `h` satisfying
`Sieve.bind S R h`. -/
abbrev BindStruct (S : Presieve X) (R : ∀ ⦃Y⦄ ⦃f : Y ⟶ X⦄, S f → Sieve Y)
    {Z : C} (h : Z ⟶ X) :=
  Presieve.BindStruct S (fun _ _ hf ↦ R hf) h


theorem generate_le_iff (R : Presieve X) (S : Sieve X) : generate R ≤ S ↔ R ≤ S :=
  ⟨fun H _ _ hg => H _ ⟨_, 𝟙 _, _, hg, id_comp _⟩, fun ss Y f => by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      R : CategoryTheory.Presieve X
      S : CategoryTheory.Sieve X
      ss : LE.le R S.arrows
      Y : C
      f : Quiver.Hom Y X
      ⊢ (CategoryTheory.Sieve.generate R).arrows f → S.arrows f
    -/
    rintro ⟨Z, f, g, hg, rfl⟩
    /-
      case intro.intro.intro.intro
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      R : CategoryTheory.Presieve X
      S : CategoryTheory.Sieve X
      ss : LE.le R S.arrows
      Y Z : C
      f : Quiver.Hom Y Z
      g : Quiver.Hom Z X
      hg : R g
      ⊢ S.arrows (CategoryTheory.CategoryStruct.comp f g)
    -/
    exact S.downward_closed (ss Z hg) f⟩
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-07-13")] alias sets_iff_generate := generate_le_iff


/-- Show that there is a galois insertion (generate, set_over). -/
def giGenerate : GaloisInsertion (generate : Presieve X → Sieve X) arrows where
  gc := generate_le_iff
  choice 𝒢 _ := generate 𝒢
  choice_eq _ _ := rfl
  le_l_u _ _ _ hf := ⟨_, 𝟙 _, _, hf, id_comp _⟩


theorem le_generate (R : Presieve X) : R ≤ generate R :=
  giGenerate.gc.le_u_l R


@[simp]
theorem generate_sieve (S : Sieve X) : generate S = S :=
  giGenerate.l_u_eq S


/-- If the identity arrow is in a sieve, the sieve is maximal. -/
theorem id_mem_iff_eq_top : S (𝟙 X) ↔ S = ⊤ :=
                                       /-
                                         C : Type u₁
                                         inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                         X : C
                                         S : CategoryTheory.Sieve X
                                         h : S.arrows (CategoryTheory.CategoryStruct.id X)
                                         Y : C
                                         f : Quiver.Hom Y X
                                         x✝ : Top.top.arrows f
                                         ⊢ S.arrows f
                                       -/
  ⟨fun h => top_unique fun Y f _ => by simpa using downward_closed _ h f, fun h => h.symm ▸ trivial⟩
                                       /-
                                         🎉 no goals
                                       -/


/-- If an arrow set contains a split epi, it generates the maximal sieve. -/
theorem generate_of_contains_isSplitEpi {R : Presieve X} (f : Y ⟶ X) [IsSplitEpi f] (hf : R f) :
    generate R = ⊤ := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    R : CategoryTheory.Presieve X
    f : Quiver.Hom Y X
    inst✝ : CategoryTheory.IsSplitEpi f
    hf : R f
    ⊢ Eq (CategoryTheory.Sieve.generate R) Top.top
  -/
  rw [← id_mem_iff_eq_top]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    R : CategoryTheory.Presieve X
    f : Quiver.Hom Y X
    inst✝ : CategoryTheory.IsSplitEpi f
    hf : R f
    ⊢ (CategoryTheory.Sieve.generate R).arrows (CategoryTheory.CategoryStruct.id X)
  -/
  exact ⟨_, section_ f, f, hf, by simp⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem generate_of_singleton_isSplitEpi (f : Y ⟶ X) [IsSplitEpi f] :
    generate (Presieve.singleton f) = ⊤ :=
  generate_of_contains_isSplitEpi f (Presieve.singleton_self _)


@[simp]
theorem generate_top : generate (⊤ : Presieve X) = ⊤ :=
  generate_of_contains_isSplitEpi (𝟙 _) ⟨⟩


@[simp]
lemma comp_mem_iff (i : X ⟶ Y) (f : Y ⟶ Z) [IsIso i] (S : Sieve Z) :
    S (i ≫ f) ↔ S f := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    i : Quiver.Hom X Y
    f : Quiver.Hom Y Z
    inst✝ : CategoryTheory.IsIso i
    S : CategoryTheory.Sieve Z
    ⊢ Iff (S.arrows (CategoryTheory.CategoryStruct.comp i f)) (S.arrows f)
  -/
  refine ⟨fun H ↦ ?_, fun H ↦ S.downward_closed H _⟩
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    i : Quiver.Hom X Y
    f : Quiver.Hom Y Z
    inst✝ : CategoryTheory.IsIso i
    S : CategoryTheory.Sieve Z
    H : S.arrows (CategoryTheory.CategoryStruct.comp i f)
    ⊢ S.arrows f
  -/
  convert S.downward_closed H (inv i)
  /-
    case h.e'_6
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    i : Quiver.Hom X Y
    f : Quiver.Hom Y Z
    inst✝ : CategoryTheory.IsIso i
    S : CategoryTheory.Sieve Z
    H : S.arrows (CategoryTheory.CategoryStruct.comp i f)
    ⊢ Eq f (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv i) (CategoryThe …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The sieve of `X` generated by family of morphisms `Y i ⟶ X`. -/
abbrev ofArrows : Sieve X := generate (Presieve.ofArrows Y f)


lemma ofArrows_mk (i : I) : ofArrows Y f (f i) :=
                      /-
                        C : Type u₁
                        inst✝ : CategoryTheory.Category.{v₁, u₁} C
                        I : Type u_1
                        X : C
                        Y : I → C
                        f : (i : I) → Quiver.Hom (Y i) X
                        i : I
                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (Y  …
                      -/
  ⟨_, 𝟙 _, _, ⟨i⟩, by simp⟩
                      /-
                        🎉 no goals
                      -/


lemma mem_ofArrows_iff {W : C} (g : W ⟶ X) :
    ofArrows Y f g ↔ ∃ (i : I) (a : W ⟶ Y i), g = a ≫ f i := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    I : Type u_1
    X : C
    Y : I → C
    f : (i : I) → Quiver.Hom (Y i) X
    W : C
    g : Quiver.Hom W X
    ⊢ Iff ((CategoryTheory.Sieve.ofArrows Y f).arrows g) (Exists fun i => Exists f …
  -/
  constructor
    /-
      case mp
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      I : Type u_1
      X : C
      Y : I → C
      f : (i : I) → Quiver.Hom (Y i) X
      W : C
      g : Quiver.Hom W X
      ⊢ (CategoryTheory.Sieve.ofArrows Y f).arrows g → Exists fun i => Exists fun a  …
    -/
  · rintro ⟨T, a, b, ⟨i⟩, rfl⟩
    /-
      case mp.intro.intro.intro.intro.mk
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      I : Type u_1
      X : C
      Y✝ : I → C
      f : (i : I) → Quiver.Hom (Y✝ i) X
      W Y : C
      i : I
      a : Quiver.Hom W (Y✝ i)
      ⊢ Exists fun i_1 => Exists fun a_1 => Eq (CategoryTheory.CategoryStruct.comp a …
    -/
    exact ⟨i, a, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      I : Type u_1
      X : C
      Y : I → C
      f : (i : I) → Quiver.Hom (Y i) X
      W : C
      g : Quiver.Hom W X
      ⊢ (Exists fun i => Exists fun a => Eq g (CategoryTheory.CategoryStruct.comp a  …
    -/
  · rintro ⟨i, a, rfl⟩
    /-
      case mpr.intro.intro
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      I : Type u_1
      X : C
      Y : I → C
      f : (i : I) → Quiver.Hom (Y i) X
      W : C
      i : I
      a : Quiver.Hom W (Y i)
      ⊢ (CategoryTheory.Sieve.ofArrows Y f).arrows (CategoryTheory.CategoryStruct.co …
    -/
    apply downward_closed _ (ofArrows_mk Y f i)
    /-
      🎉 no goals
    -/


include hg in
lemma ofArrows.exists : ∃ (i : I) (h : W ⟶ Y i), g = h ≫ f i := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    I : Type u_1
    X : C
    Y : I → C
    f : (i : I) → Quiver.Hom (Y i) X
    W : C
    g : Quiver.Hom W X
    hg : (CategoryTheory.Sieve.ofArrows Y f).arrows g
    ⊢ Exists fun i => Exists fun h => Eq g (CategoryTheory.CategoryStruct.comp h ( …
  -/
  obtain ⟨_, h, _, H, rfl⟩ := hg
  /-
    case intro.intro.intro.intro
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    I : Type u_1
    X : C
    Y : I → C
    f : (i : I) → Quiver.Hom (Y i) X
    W w✝¹ : C
    h : Quiver.Hom W w✝¹
    w✝ : Quiver.Hom w✝¹ X
    H : CategoryTheory.Presieve.ofArrows Y f w✝
    ⊢ Exists fun i => Exists fun h_1 => Eq (CategoryTheory.CategoryStruct.comp h w …
  -/
  cases' H with i
  /-
    case intro.intro.intro.intro.mk
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    I : Type u_1
    X : C
    Y : I → C
    f : (i : I) → Quiver.Hom (Y i) X
    W : C
    i : I
    h : Quiver.Hom W (Y i)
    ⊢ Exists fun i_1 => Exists fun h_1 => Eq (CategoryTheory.CategoryStruct.comp h …
  -/
  exact ⟨i, h, rfl⟩
  /-
    🎉 no goals
  -/


/-- When `hg : Sieve.ofArrows Y f g`, this is a choice of `i` such that `g`
factors through `f i`. -/
noncomputable def ofArrows.i : I := (ofArrows.exists hg).choose


/-- When `hg : Sieve.ofArrows Y f g`, this is a morphism `h : W ⟶ Y (i hg)` such
that `h ≫ f (i hg) = g`. -/
noncomputable def ofArrows.h : W ⟶ Y (i hg) := (ofArrows.exists hg).choose_spec.choose


@[reassoc (attr := simp)]
lemma ofArrows.fac : h hg ≫ f (i hg) = g :=
  (ofArrows.exists hg).choose_spec.choose_spec.symm


/-- The sieve generated by two morphisms. -/
abbrev ofTwoArrows {U V X : C} (i : U ⟶ X) (j : V ⟶ X) : Sieve X :=
  Sieve.ofArrows (Y := pairFunction U V) (fun k ↦ WalkingPair.casesOn k i j)


/-- The sieve of `X : C` that is generated by a family of objects `Y : I → C`:
it consists of morphisms to `X` which factor through at least one of the `Y i`. -/
def ofObjects {I : Type*} (Y : I → C) (X : C) : Sieve X where
  arrows Z _ := ∃ (i : I), Nonempty (Z ⟶ Y i)
  downward_closed := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X✝ Y✝ Z : C
      f : Quiver.Hom Y✝ X✝
      S R : CategoryTheory.Sieve X✝
      I : Type u_1
      Y : I → C
      X : C
      ⊢ ∀ {Y_1 Z : C} {f : Quiver.Hom Y_1 X}, (fun Z x => Exists fun i => Nonempty ( …
    -/
    rintro Z₁ Z₂ p ⟨i, ⟨f⟩⟩ g
    /-
      case intro.intro
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X✝ Y✝ Z : C
      f✝ : Quiver.Hom Y✝ X✝
      S R : CategoryTheory.Sieve X✝
      I : Type u_1
      Y : I → C
      X Z₁ Z₂ : C
      p : Quiver.Hom Z₁ X
      i : I
      f : Quiver.Hom Z₁ (Y i)
      g : Quiver.Hom Z₂ Z₁
      ⊢ Exists fun i => Nonempty (Quiver.Hom Z₂ (Y i))
    -/
    exact ⟨i, ⟨g ≫ f⟩⟩
    /-
      🎉 no goals
    -/


lemma mem_ofObjects_iff {I : Type*} (Y : I → C) {Z X : C} (g : Z ⟶ X) :
                                                          /-
                                                            C : Type u₁
                                                            inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                                            I : Type u_1
                                                            Y : I → C
                                                            Z X : C
                                                            g : Quiver.Hom Z X
                                                            ⊢ Iff ((CategoryTheory.Sieve.ofObjects Y X).arrows g) (Exists fun i => Nonempt …
                                                          -/
    ofObjects Y X g ↔ ∃ (i : I), Nonempty (Z ⟶ Y i) := by rfl
                                                          /-
                                                            🎉 no goals
                                                          -/


lemma ofArrows_le_ofObjects
    {I : Type*} (Y : I → C) {X : C} (f : ∀ i, Y i ⟶ X) :
    Sieve.ofArrows Y f ≤ Sieve.ofObjects Y X := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    I : Type u_1
    Y : I → C
    X : C
    f : (i : I) → Quiver.Hom (Y i) X
    ⊢ LE.le (CategoryTheory.Sieve.ofArrows Y f) (CategoryTheory.Sieve.ofObjects Y X)
  -/
  intro W g hg
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    I : Type u_1
    Y : I → C
    X : C
    f : (i : I) → Quiver.Hom (Y i) X
    W : C
    g : Quiver.Hom W X
    hg : (CategoryTheory.Sieve.ofArrows Y f).arrows g
    ⊢ (CategoryTheory.Sieve.ofObjects Y X).arrows g
  -/
  rw [mem_ofArrows_iff] at hg
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    I : Type u_1
    Y : I → C
    X : C
    f : (i : I) → Quiver.Hom (Y i) X
    W : C
    g : Quiver.Hom W X
    hg : Exists fun i => Exists fun a => Eq g (CategoryTheory.CategoryStruct.comp  …
    ⊢ (CategoryTheory.Sieve.ofObjects Y X).arrows g
  -/
  obtain ⟨i, a, rfl⟩ := hg
  /-
    case intro.intro
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    I : Type u_1
    Y : I → C
    X : C
    f : (i : I) → Quiver.Hom (Y i) X
    W : C
    i : I
    a : Quiver.Hom W (Y i)
    ⊢ (CategoryTheory.Sieve.ofObjects Y X).arrows (CategoryTheory.CategoryStruct.c …
  -/
  exact ⟨i, ⟨a⟩⟩
  /-
    🎉 no goals
  -/


lemma ofArrows_eq_ofObjects {X : C} (hX : IsTerminal X)
    {I : Type*} (Y : I → C) (f : ∀ i, Y i ⟶ X) :
    ofArrows Y f = ofObjects Y X := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    hX : CategoryTheory.Limits.IsTerminal X
    I : Type u_1
    Y : I → C
    f : (i : I) → Quiver.Hom (Y i) X
    ⊢ Eq (CategoryTheory.Sieve.ofArrows Y f) (CategoryTheory.Sieve.ofObjects Y X)
  -/
  refine le_antisymm (ofArrows_le_ofObjects Y f) (fun W g => ?_)
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    hX : CategoryTheory.Limits.IsTerminal X
    I : Type u_1
    Y : I → C
    f : (i : I) → Quiver.Hom (Y i) X
    W : C
    g : Quiver.Hom W X
    ⊢ (CategoryTheory.Sieve.ofObjects Y X).arrows g → (CategoryTheory.Sieve.ofArro …
  -/
  rw [mem_ofArrows_iff, mem_ofObjects_iff]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    hX : CategoryTheory.Limits.IsTerminal X
    I : Type u_1
    Y : I → C
    f : (i : I) → Quiver.Hom (Y i) X
    W : C
    g : Quiver.Hom W X
    ⊢ (Exists fun i => Nonempty (Quiver.Hom W (Y i))) → Exists fun i => Exists fun …
  -/
  rintro ⟨i, ⟨h⟩⟩
  /-
    case intro.intro
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    hX : CategoryTheory.Limits.IsTerminal X
    I : Type u_1
    Y : I → C
    f : (i : I) → Quiver.Hom (Y i) X
    W : C
    g : Quiver.Hom W X
    i : I
    h : Quiver.Hom W (Y i)
    ⊢ Exists fun i => Exists fun a => Eq g (CategoryTheory.CategoryStruct.comp a ( …
  -/
  exact ⟨i, h, hX.hom_ext _ _⟩
  /-
    🎉 no goals
  -/


/-- Given a morphism `h : Y ⟶ X`, send a sieve S on X to a sieve on Y
    as the inverse image of S with `_ ≫ h`.
    That is, `Sieve.pullback S h := (≫ h) '⁻¹ S`. -/
@[simps]
def pullback (h : Y ⟶ X) (S : Sieve X) : Sieve Y where
  arrows _ sl := S (sl ≫ h)
                          /-
                            C : Type u₁
                            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                            D : Type u₂
                            inst✝ : CategoryTheory.Category.{v₂, u₂} D
                            F : CategoryTheory.Functor C D
                            X Y Z : C
                            f : Quiver.Hom Y X
                            S✝ R : CategoryTheory.Sieve X
                            h : Quiver.Hom Y X
                            S : CategoryTheory.Sieve X
                            Y✝ Z✝ : C
                            f✝ : Quiver.Hom Y✝ Y
                            g : (fun x sl => S.arrows (CategoryTheory.CategoryStruct.comp sl h)) Y✝ f✝
                            ⊢ ∀ (g : Quiver.Hom Z✝ Y✝), (fun x sl => S.arrows (CategoryTheory.CategoryStru …
                          -/
  downward_closed g := by simp [g]
                          /-
                            🎉 no goals
                          -/


@[simp]
                                                 /-
                                                   C : Type u₁
                                                   inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                                   X : C
                                                   S : CategoryTheory.Sieve X
                                                   ⊢ Eq (CategoryTheory.Sieve.pullback (CategoryTheory.CategoryStruct.id X) S) S
                                                 -/
theorem pullback_id : S.pullback (𝟙 _) = S := by simp [Sieve.ext_iff]
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
theorem pullback_top {f : Y ⟶ X} : (⊤ : Sieve X).pullback f = ⊤ :=
  top_unique fun _ _ => id


theorem pullback_comp {f : Y ⟶ X} {g : Z ⟶ Y} (S : Sieve X) :
                                                         /-
                                                           C : Type u₁
                                                           inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                                           X Y Z : C
                                                           f : Quiver.Hom Y X
                                                           g : Quiver.Hom Z Y
                                                           S : CategoryTheory.Sieve X
                                                           ⊢ Eq (CategoryTheory.Sieve.pullback (CategoryTheory.CategoryStruct.comp g f) S …
                                                         -/
    S.pullback (g ≫ f) = (S.pullback f).pullback g := by simp [Sieve.ext_iff]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp]
theorem pullback_inter {f : Y ⟶ X} (S R : Sieve X) :
                                                           /-
                                                             C : Type u₁
                                                             inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                                             X Y : C
                                                             f : Quiver.Hom Y X
                                                             S R : CategoryTheory.Sieve X
                                                             ⊢ Eq (CategoryTheory.Sieve.pullback f (Min.min S R)) (Min.min (CategoryTheory. …
                                                           -/
    (S ⊓ R).pullback f = S.pullback f ⊓ R.pullback f := by simp [Sieve.ext_iff]
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem pullback_eq_top_iff_mem (f : Y ⟶ X) : S f ↔ S.pullback f = ⊤ := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    S : CategoryTheory.Sieve X
    f : Quiver.Hom Y X
    ⊢ Iff (S.arrows f) (Eq (CategoryTheory.Sieve.pullback f S) Top.top)
  -/
  rw [← id_mem_iff_eq_top, pullback_apply, id_comp]
  /-
    🎉 no goals
  -/


theorem pullback_eq_top_of_mem (S : Sieve X) {f : Y ⟶ X} : S f → S.pullback f = ⊤ :=
  (pullback_eq_top_iff_mem f).1


lemma pullback_ofObjects_eq_top
    {I : Type*} (Y : I → C) {X : C} {i : I} (g : X ⟶ Y i) :
    ofObjects Y X = ⊤ := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    I : Type u_1
    Y : I → C
    X : C
    i : I
    g : Quiver.Hom X (Y i)
    ⊢ Eq (CategoryTheory.Sieve.ofObjects Y X) Top.top
  -/
  ext Z h
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    I : Type u_1
    Y : I → C
    X : C
    i : I
    g : Quiver.Hom X (Y i)
    Z : C
    h : Quiver.Hom Z X
    ⊢ Iff ((CategoryTheory.Sieve.ofObjects Y X).arrows h) (Top.top.arrows h)
  -/
  simp only [top_apply, iff_true]
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    I : Type u_1
    Y : I → C
    X : C
    i : I
    g : Quiver.Hom X (Y i)
    Z : C
    h : Quiver.Hom Z X
    ⊢ (CategoryTheory.Sieve.ofObjects Y X).arrows h
  -/
  rw [mem_ofObjects_iff ]
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    I : Type u_1
    Y : I → C
    X : C
    i : I
    g : Quiver.Hom X (Y i)
    Z : C
    h : Quiver.Hom Z X
    ⊢ Exists fun i => Nonempty (Quiver.Hom Z (Y i))
  -/
  exact ⟨i, ⟨h ≫ g⟩⟩
  /-
    🎉 no goals
  -/


/-- Push a sieve `R` on `Y` forward along an arrow `f : Y ⟶ X`: `gf : Z ⟶ X` is in the sieve if `gf`
factors through some `g : Z ⟶ Y` which is in `R`.
-/
@[simps]
def pushforward (f : Y ⟶ X) (R : Sieve Y) : Sieve X where
  arrows _ gf := ∃ g, g ≫ f = gf ∧ R g
                                                   /-
                                                     C : Type u₁
                                                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                     D : Type u₂
                                                     inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                     F : CategoryTheory.Functor C D
                                                     X Y Z : C
                                                     f✝¹ : Quiver.Hom Y X
                                                     S R✝ : CategoryTheory.Sieve X
                                                     f : Quiver.Hom Y X
                                                     R : CategoryTheory.Sieve Y
                                                     Y✝ Z✝ : C
                                                     f✝ : Quiver.Hom Y✝ X
                                                     x✝ : (fun x gf => Exists fun g => And (Eq (CategoryTheory.CategoryStruct.comp  …
                                                     h : Quiver.Hom Z✝ Y✝
                                                     j : Quiver.Hom Y✝ Y
                                                     k : Eq (CategoryTheory.CategoryStruct.comp j f) f✝
                                                     z : R.arrows j
                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp h …
                                                   -/
                                                   /-
                                                     🎉 no goals
                                                   -/
  downward_closed := fun ⟨j, k, z⟩ h => ⟨h ≫ j, by simp [k], by simp [z]⟩
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem pushforward_apply_comp {R : Sieve Y} {Z : C} {g : Z ⟶ Y} (hg : R g) (f : Y ⟶ X) :
    R.pushforward f (g ≫ f) :=
  ⟨g, rfl, hg⟩


theorem pushforward_comp {f : Y ⟶ X} {g : Z ⟶ Y} (R : Sieve Z) :
    R.pushforward (g ≫ f) = (R.pushforward g).pushforward f :=
  Sieve.ext fun W h =>
                                      /-
                                        C : Type u₁
                                        inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                        X Y Z : C
                                        f : Quiver.Hom Y X
                                        g : Quiver.Hom Z Y
                                        R : CategoryTheory.Sieve Z
                                        W : C
                                        h : Quiver.Hom W X
                                        x✝ : (CategoryTheory.Sieve.pushforward (CategoryTheory.CategoryStruct.comp g f …
                                        f₁ : Quiver.Hom W Z
                                        hq : Eq (CategoryTheory.CategoryStruct.comp f₁ (CategoryTheory.CategoryStruct. …
                                        hf₁ : R.arrows f₁
                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
                                      -/
    ⟨fun ⟨f₁, hq, hf₁⟩ => ⟨f₁ ≫ g, by simpa, f₁, rfl, hf₁⟩, fun ⟨y, hy, z, hR, hz⟩ =>
                                      /-
                                        🎉 no goals
                                      -/
             /-
               C : Type u₁
               inst✝ : CategoryTheory.Category.{v₁, u₁} C
               X Y Z : C
               f : Quiver.Hom Y X
               g : Quiver.Hom Z Y
               R : CategoryTheory.Sieve Z
               W : C
               h : Quiver.Hom W X
               x✝ : (CategoryTheory.Sieve.pushforward f (CategoryTheory.Sieve.pushforward g R …
               y : Quiver.Hom W Y
               hy : Eq (CategoryTheory.CategoryStruct.comp y f) h
               z : Quiver.Hom W Z
               hR : Eq (CategoryTheory.CategoryStruct.comp z g) y
               hz : R.arrows z
               ⊢ And (Eq (CategoryTheory.CategoryStruct.comp z (CategoryTheory.CategoryStruct …
             -/
      ⟨z, by rw [← Category.assoc, hR]; tauto⟩⟩
                                        /-
                                          🎉 no goals
                                        -/


theorem galoisConnection (f : Y ⟶ X) : GaloisConnection (Sieve.pushforward f) (Sieve.pullback f) :=
  fun _ _ => ⟨fun hR _ g hg => hR _ ⟨g, rfl, hg⟩, fun hS _ _ ⟨h, hg, hh⟩ => hg ▸ hS h hh⟩


theorem pullback_monotone (f : Y ⟶ X) : Monotone (Sieve.pullback f) :=
  (galoisConnection f).monotone_u


theorem pushforward_monotone (f : Y ⟶ X) : Monotone (Sieve.pushforward f) :=
  (galoisConnection f).monotone_l


theorem le_pushforward_pullback (f : Y ⟶ X) (R : Sieve Y) : R ≤ (R.pushforward f).pullback f :=
  (galoisConnection f).le_u_l _


theorem pullback_pushforward_le (f : Y ⟶ X) (R : Sieve X) : (R.pullback f).pushforward f ≤ R :=
  (galoisConnection f).l_u_le _


theorem pushforward_union {f : Y ⟶ X} (S R : Sieve Y) :
    (S ⊔ R).pushforward f = S.pushforward f ⊔ R.pushforward f :=
  (galoisConnection f).l_sup


theorem pushforward_le_bind_of_mem (S : Presieve X) (R : ∀ ⦃Y : C⦄ ⦃f : Y ⟶ X⦄, S f → Sieve Y)
    (f : Y ⟶ X) (h : S f) : (R h).pushforward f ≤ bind S R := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    S : CategoryTheory.Presieve X
    R : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → S f → CategoryTheory.Sieve Y
    f : Quiver.Hom Y X
    h : S f
    ⊢ LE.le (CategoryTheory.Sieve.pushforward f (R h)) (CategoryTheory.Sieve.bind  …
  -/
  rintro Z _ ⟨g, rfl, hg⟩
  /-
    case intro.intro
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    S : CategoryTheory.Presieve X
    R : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → S f → CategoryTheory.Sieve Y
    f : Quiver.Hom Y X
    h : S f
    Z : C
    g : Quiver.Hom Z Y
    hg : (R h).arrows g
    ⊢ (CategoryTheory.Sieve.bind S R).arrows (CategoryTheory.CategoryStruct.comp g …
  -/
  exact ⟨_, g, f, h, hg, rfl⟩
  /-
    🎉 no goals
  -/


theorem le_pullback_bind (S : Presieve X) (R : ∀ ⦃Y : C⦄ ⦃f : Y ⟶ X⦄, S f → Sieve Y) (f : Y ⟶ X)
    (h : S f) : R h ≤ (bind S R).pullback f := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    S : CategoryTheory.Presieve X
    R : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → S f → CategoryTheory.Sieve Y
    f : Quiver.Hom Y X
    h : S f
    ⊢ LE.le (R h) (CategoryTheory.Sieve.pullback f (CategoryTheory.Sieve.bind S R))
  -/
  rw [← galoisConnection f]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    S : CategoryTheory.Presieve X
    R : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y X⦄ → S f → CategoryTheory.Sieve Y
    f : Quiver.Hom Y X
    h : S f
    ⊢ LE.le (CategoryTheory.Sieve.pushforward f (R h)) (CategoryTheory.Sieve.bind  …
  -/
  apply pushforward_le_bind_of_mem
  /-
    🎉 no goals
  -/


/-- If `f` is a monomorphism, the pushforward-pullback adjunction on sieves is coreflective. -/
def galoisCoinsertionOfMono (f : Y ⟶ X) [Mono f] :
    GaloisCoinsertion (Sieve.pushforward f) (Sieve.pullback f) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y Z : C
    f✝ : Quiver.Hom Y X
    S R : CategoryTheory.Sieve X
    f : Quiver.Hom Y X
    inst✝ : CategoryTheory.Mono f
    ⊢ GaloisCoinsertion (CategoryTheory.Sieve.pushforward f) (CategoryTheory.Sieve …
  -/
  apply (galoisConnection f).toGaloisCoinsertion
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y Z : C
    f✝ : Quiver.Hom Y X
    S R : CategoryTheory.Sieve X
    f : Quiver.Hom Y X
    inst✝ : CategoryTheory.Mono f
    ⊢ ∀ (a : CategoryTheory.Sieve Y), LE.le (CategoryTheory.Sieve.pullback f (Cate …
  -/
  rintro S Z g ⟨g₁, hf, hg₁⟩
  /-
    case intro.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y Z✝ : C
    f✝ : Quiver.Hom Y X
    S✝ R : CategoryTheory.Sieve X
    f : Quiver.Hom Y X
    inst✝ : CategoryTheory.Mono f
    S : CategoryTheory.Sieve Y
    Z : C
    g g₁ : Quiver.Hom Z Y
    hf : Eq (CategoryTheory.CategoryStruct.comp g₁ f) (CategoryTheory.CategoryStru …
    hg₁ : S.arrows g₁
    ⊢ S.arrows g
  -/
  rw [cancel_mono f] at hf
  /-
    case intro.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y Z✝ : C
    f✝ : Quiver.Hom Y X
    S✝ R : CategoryTheory.Sieve X
    f : Quiver.Hom Y X
    inst✝ : CategoryTheory.Mono f
    S : CategoryTheory.Sieve Y
    Z : C
    g g₁ : Quiver.Hom Z Y
    hf : Eq g₁ g
    hg₁ : S.arrows g₁
    ⊢ S.arrows g
  -/
  rwa [← hf]
  /-
    🎉 no goals
  -/


/-- If `f` is a split epi, the pushforward-pullback adjunction on sieves is reflective. -/
def galoisInsertionOfIsSplitEpi (f : Y ⟶ X) [IsSplitEpi f] :
    GaloisInsertion (Sieve.pushforward f) (Sieve.pullback f) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y Z : C
    f✝ : Quiver.Hom Y X
    S R : CategoryTheory.Sieve X
    f : Quiver.Hom Y X
    inst✝ : CategoryTheory.IsSplitEpi f
    ⊢ GaloisInsertion (CategoryTheory.Sieve.pushforward f) (CategoryTheory.Sieve.p …
  -/
  apply (galoisConnection f).toGaloisInsertion
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y Z : C
    f✝ : Quiver.Hom Y X
    S R : CategoryTheory.Sieve X
    f : Quiver.Hom Y X
    inst✝ : CategoryTheory.IsSplitEpi f
    ⊢ ∀ (b : CategoryTheory.Sieve X), LE.le b (CategoryTheory.Sieve.pushforward f  …
  -/
  intro S Z g hg
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y Z✝ : C
    f✝ : Quiver.Hom Y X
    S✝ R : CategoryTheory.Sieve X
    f : Quiver.Hom Y X
    inst✝ : CategoryTheory.IsSplitEpi f
    S : CategoryTheory.Sieve X
    Z : C
    g : Quiver.Hom Z X
    hg : S.arrows g
    ⊢ (CategoryTheory.Sieve.pushforward f (CategoryTheory.Sieve.pullback f S)).arr …
  -/
  exact ⟨g ≫ section_ f, by simpa⟩
  /-
    🎉 no goals
  -/


theorem pullbackArrows_comm [HasPullbacks C] {X Y : C} (f : Y ⟶ X) (R : Presieve X) :
    Sieve.generate (R.pullbackArrows f) = (Sieve.generate R).pullback f := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    X Y : C
    f : Quiver.Hom Y X
    R : CategoryTheory.Presieve X
    ⊢ Eq (CategoryTheory.Sieve.generate (CategoryTheory.Presieve.pullbackArrows f  …
  -/
  ext W g
  /-
    case h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    X Y : C
    f : Quiver.Hom Y X
    R : CategoryTheory.Presieve X
    W : C
    g : Quiver.Hom W Y
    ⊢ Iff ((CategoryTheory.Sieve.generate (CategoryTheory.Presieve.pullbackArrows  …
  -/
  constructor
    /-
      case h.mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      X Y : C
      f : Quiver.Hom Y X
      R : CategoryTheory.Presieve X
      W : C
      g : Quiver.Hom W Y
      ⊢ (CategoryTheory.Sieve.generate (CategoryTheory.Presieve.pullbackArrows f R)) …
    -/
  · rintro ⟨_, h, k, hk, rfl⟩
    /-
      case h.mp.intro.intro.intro.intro
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      X Y : C
      f : Quiver.Hom Y X
      R : CategoryTheory.Presieve X
      W w✝ : C
      h : Quiver.Hom W w✝
      k : Quiver.Hom w✝ Y
      hk : CategoryTheory.Presieve.pullbackArrows f R k
      ⊢ (CategoryTheory.Sieve.pullback f (CategoryTheory.Sieve.generate R)).arrows ( …
    -/
    cases' hk with W g hg
    /-
      case h.mp.intro.intro.intro.intro.mk
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      X Y : C
      f : Quiver.Hom Y X
      R : CategoryTheory.Presieve X
      W✝ W : C
      g : Quiver.Hom W X
      hg : R g
      h : Quiver.Hom W✝ (CategoryTheory.Limits.pullback g f)
      ⊢ (CategoryTheory.Sieve.pullback f (CategoryTheory.Sieve.generate R)).arrows ( …
    -/
    rw [Sieve.pullback_apply, assoc, ← pullback.condition, ← assoc]
    /-
      case h.mp.intro.intro.intro.intro.mk
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      X Y : C
      f : Quiver.Hom Y X
      R : CategoryTheory.Presieve X
      W✝ W : C
      g : Quiver.Hom W X
      hg : R g
      h : Quiver.Hom W✝ (CategoryTheory.Limits.pullback g f)
      ⊢ (CategoryTheory.Sieve.generate R).arrows (CategoryTheory.CategoryStruct.comp …
    -/
    exact Sieve.downward_closed _ (by exact Sieve.le_generate R W hg) (h ≫ pullback.fst g f)
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      X Y : C
      f : Quiver.Hom Y X
      R : CategoryTheory.Presieve X
      W : C
      g : Quiver.Hom W Y
      ⊢ (CategoryTheory.Sieve.pullback f (CategoryTheory.Sieve.generate R)).arrows g …
    -/
  · rintro ⟨W, h, k, hk, comm⟩
    /-
      case h.mpr.intro.intro.intro.intro
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      X Y : C
      f : Quiver.Hom Y X
      R : CategoryTheory.Presieve X
      W✝ : C
      g : Quiver.Hom W✝ Y
      W : C
      h : Quiver.Hom W✝ W
      k : Quiver.Hom W X
      hk : R k
      comm : Eq (CategoryTheory.CategoryStruct.comp h k) (CategoryTheory.CategoryStr …
      ⊢ (CategoryTheory.Sieve.generate (CategoryTheory.Presieve.pullbackArrows f R)) …
    -/
    exact ⟨_, _, _, Presieve.pullbackArrows.mk _ _ hk, pullback.lift_snd _ _ comm⟩
    /-
      🎉 no goals
    -/


/--
If `R` is a sieve, then the `CategoryTheory.Presieve.functorPullback` of `R` is actually a sieve.
-/
@[simps]
def functorPullback (R : Sieve (F.obj X)) : Sieve X where
  arrows := Presieve.functorPullback F R
  downward_closed := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y Z : C
      f : Quiver.Hom Y X
      S R✝ : CategoryTheory.Sieve X
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      R : CategoryTheory.Sieve (F.obj X)
      ⊢ ∀ {Y Z : C} {f : Quiver.Hom Y X}, CategoryTheory.Presieve.functorPullback F  …
    -/
    intro _ _ f hf g
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y Z : C
      f✝ : Quiver.Hom Y X
      S R✝ : CategoryTheory.Sieve X
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      R : CategoryTheory.Sieve (F.obj X)
      Y✝ Z✝ : C
      f : Quiver.Hom Y✝ X
      hf : CategoryTheory.Presieve.functorPullback F R.arrows f
      g : Quiver.Hom Z✝ Y✝
      ⊢ CategoryTheory.Presieve.functorPullback F R.arrows (CategoryTheory.CategoryS …
    -/
    unfold Presieve.functorPullback
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y Z : C
      f✝ : Quiver.Hom Y X
      S R✝ : CategoryTheory.Sieve X
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      R : CategoryTheory.Sieve (F.obj X)
      Y✝ Z✝ : C
      f : Quiver.Hom Y✝ X
      hf : CategoryTheory.Presieve.functorPullback F R.arrows f
      g : Quiver.Hom Z✝ Y✝
      ⊢ R.arrows (F.map (CategoryTheory.CategoryStruct.comp g f))
    -/
    rw [F.map_comp]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y Z : C
      f✝ : Quiver.Hom Y X
      S R✝ : CategoryTheory.Sieve X
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      R : CategoryTheory.Sieve (F.obj X)
      Y✝ Z✝ : C
      f : Quiver.Hom Y✝ X
      hf : CategoryTheory.Presieve.functorPullback F R.arrows f
      g : Quiver.Hom Z✝ Y✝
      ⊢ R.arrows (CategoryTheory.CategoryStruct.comp (F.map g) (F.map f))
    -/
    exact R.downward_closed hf (F.map g)
    /-
      🎉 no goals
    -/


@[simp]
theorem functorPullback_arrows (R : Sieve (F.obj X)) :
    (R.functorPullback F).arrows = R.arrows.functorPullback F :=
  rfl


@[simp]
theorem functorPullback_id (R : Sieve X) : R.functorPullback (𝟭 _) = R := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    R : CategoryTheory.Sieve X
    ⊢ Eq (CategoryTheory.Sieve.functorPullback (CategoryTheory.Functor.id C) R) R
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    R : CategoryTheory.Sieve X
    Y✝ : C
    f✝ : Quiver.Hom Y✝ X
    ⊢ Iff ((CategoryTheory.Sieve.functorPullback (CategoryTheory.Functor.id C) R). …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem functorPullback_comp (R : Sieve ((F ⋙ G).obj X)) :
    R.functorPullback (F ⋙ G) = (R.functorPullback G).functorPullback F := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X : C
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    R : CategoryTheory.Sieve ((F.comp G).obj X)
    ⊢ Eq (CategoryTheory.Sieve.functorPullback (F.comp G) R) (CategoryTheory.Sieve …
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X : C
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    R : CategoryTheory.Sieve ((F.comp G).obj X)
    Y✝ : C
    f✝ : Quiver.Hom Y✝ X
    ⊢ Iff ((CategoryTheory.Sieve.functorPullback (F.comp G) R).arrows f✝) ((Catego …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem functorPushforward_extend_eq {R : Presieve X} :
    (generate R).arrows.functorPushforward F = R.functorPushforward F := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X : C
    R : CategoryTheory.Presieve X
    ⊢ Eq (CategoryTheory.Presieve.functorPushforward F (CategoryTheory.Sieve.gener …
  -/
  funext Y
  /-
    case h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X : C
    R : CategoryTheory.Presieve X
    Y : D
    ⊢ Eq (CategoryTheory.Presieve.functorPushforward F (CategoryTheory.Sieve.gener …
  -/
  ext f
  /-
    case h.h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X : C
    R : CategoryTheory.Presieve X
    Y : D
    f : Quiver.Hom Y (F.obj X)
    ⊢ Iff (Membership.mem (CategoryTheory.Presieve.functorPushforward F (CategoryT …
  -/
  constructor
    /-
      case h.h.mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X : C
      R : CategoryTheory.Presieve X
      Y : D
      f : Quiver.Hom Y (F.obj X)
      ⊢ Membership.mem (CategoryTheory.Presieve.functorPushforward F (CategoryTheory …
    -/
  · rintro ⟨X', g, f', ⟨X'', g', f'', h₁, rfl⟩, rfl⟩
    /-
      case h.h.mp.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X : C
      R : CategoryTheory.Presieve X
      Y : D
      X' : C
      f' : Quiver.Hom Y (F.obj X')
      X'' : C
      g' : Quiver.Hom X' X''
      f'' : Quiver.Hom X'' X
      h₁ : R f''
      ⊢ Membership.mem (CategoryTheory.Presieve.functorPushforward F R) (CategoryThe …
    -/
    exact ⟨X'', f'', f' ≫ F.map g', h₁, by simp⟩
    /-
      🎉 no goals
    -/
    /-
      case h.h.mpr
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X : C
      R : CategoryTheory.Presieve X
      Y : D
      f : Quiver.Hom Y (F.obj X)
      ⊢ Membership.mem (CategoryTheory.Presieve.functorPushforward F R) f → Membersh …
    -/
  · rintro ⟨X', g, f', h₁, h₂⟩
    /-
      case h.h.mpr.intro.intro.intro.intro
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X : C
      R : CategoryTheory.Presieve X
      Y : D
      f : Quiver.Hom Y (F.obj X)
      X' : C
      g : Quiver.Hom X' X
      f' : Quiver.Hom Y (F.obj X')
      h₁ : R g
      h₂ : Eq f (CategoryTheory.CategoryStruct.comp f' (F.map g))
      ⊢ Membership.mem (CategoryTheory.Presieve.functorPushforward F (CategoryTheory …
    -/
    exact ⟨X', g, f', le_generate R _ h₁, h₂⟩
    /-
      🎉 no goals
    -/


/-- The sieve generated by the image of `R` under `F`. -/
@[simps]
def functorPushforward (R : Sieve X) : Sieve (F.obj X) where
  arrows := R.arrows.functorPushforward F
  downward_closed := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y Z : C
      f : Quiver.Hom Y X
      S R✝ : CategoryTheory.Sieve X
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      R : CategoryTheory.Sieve X
      ⊢ ∀ {Y Z : D} {f : Quiver.Hom Y (F.obj X)}, CategoryTheory.Presieve.functorPus …
    -/
    intro _ _ f h g
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y Z : C
      f✝ : Quiver.Hom Y X
      S R✝ : CategoryTheory.Sieve X
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      R : CategoryTheory.Sieve X
      Y✝ Z✝ : D
      f : Quiver.Hom Y✝ (F.obj X)
      h : CategoryTheory.Presieve.functorPushforward F R.arrows f
      g : Quiver.Hom Z✝ Y✝
      ⊢ CategoryTheory.Presieve.functorPushforward F R.arrows (CategoryTheory.Catego …
    -/
    obtain ⟨X, α, β, hα, rfl⟩ := h
    /-
      case intro.intro.intro.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X✝ Y Z : C
      f : Quiver.Hom Y X✝
      S R✝ : CategoryTheory.Sieve X✝
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      R : CategoryTheory.Sieve X✝
      Y✝ Z✝ : D
      g : Quiver.Hom Z✝ Y✝
      X : C
      α : Quiver.Hom X X✝
      β : Quiver.Hom Y✝ (F.obj X)
      hα : R.arrows α
      ⊢ CategoryTheory.Presieve.functorPushforward F R.arrows (CategoryTheory.Catego …
    -/
    exact ⟨X, α, g ≫ β, hα, by simp⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem functorPushforward_id (R : Sieve X) : R.functorPushforward (𝟭 _) = R := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    R : CategoryTheory.Sieve X
    ⊢ Eq (CategoryTheory.Sieve.functorPushforward (CategoryTheory.Functor.id C) R) R
  -/
  ext X f
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X✝ : C
    R : CategoryTheory.Sieve X✝
    X : C
    f : Quiver.Hom X ((CategoryTheory.Functor.id C).obj X✝)
    ⊢ Iff ((CategoryTheory.Sieve.functorPushforward (CategoryTheory.Functor.id C)  …
  -/
  constructor
    /-
      case h.mp
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X✝ : C
      R : CategoryTheory.Sieve X✝
      X : C
      f : Quiver.Hom X ((CategoryTheory.Functor.id C).obj X✝)
      ⊢ (CategoryTheory.Sieve.functorPushforward (CategoryTheory.Functor.id C) R).ar …
    -/
  · intro hf
    /-
      case h.mp
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X✝ : C
      R : CategoryTheory.Sieve X✝
      X : C
      f : Quiver.Hom X ((CategoryTheory.Functor.id C).obj X✝)
      hf : (CategoryTheory.Sieve.functorPushforward (CategoryTheory.Functor.id C) R) …
      ⊢ R.arrows f
    -/
    obtain ⟨X, g, h, hg, rfl⟩ := hf
    /-
      case h.mp.intro.intro.intro.intro
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X✝¹ : C
      R : CategoryTheory.Sieve X✝¹
      X✝ X : C
      g : Quiver.Hom X X✝¹
      h : Quiver.Hom X✝ ((CategoryTheory.Functor.id C).obj X)
      hg : R.arrows g
      ⊢ R.arrows (CategoryTheory.CategoryStruct.comp h ((CategoryTheory.Functor.id C …
    -/
    exact R.downward_closed hg h
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X✝ : C
      R : CategoryTheory.Sieve X✝
      X : C
      f : Quiver.Hom X ((CategoryTheory.Functor.id C).obj X✝)
      ⊢ R.arrows f → (CategoryTheory.Sieve.functorPushforward (CategoryTheory.Functo …
    -/
  · intro hf
    /-
      case h.mpr
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X✝ : C
      R : CategoryTheory.Sieve X✝
      X : C
      f : Quiver.Hom X ((CategoryTheory.Functor.id C).obj X✝)
      hf : R.arrows f
      ⊢ (CategoryTheory.Sieve.functorPushforward (CategoryTheory.Functor.id C) R).ar …
    -/
    exact ⟨X, f, 𝟙 _, hf, by simp⟩
    /-
      🎉 no goals
    -/


theorem functorPushforward_comp (R : Sieve X) :
    R.functorPushforward (F ⋙ G) = (R.functorPushforward F).functorPushforward G := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X : C
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    R : CategoryTheory.Sieve X
    ⊢ Eq (CategoryTheory.Sieve.functorPushforward (F.comp G) R) (CategoryTheory.Si …
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X : C
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    R : CategoryTheory.Sieve X
    Y✝ : E
    f✝ : Quiver.Hom Y✝ ((F.comp G).obj X)
    ⊢ Iff ((CategoryTheory.Sieve.functorPushforward (F.comp G) R).arrows f✝) ((Cat …
  -/
  simp [R.arrows.functorPushforward_comp F G]
  /-
    🎉 no goals
  -/


theorem functor_galoisConnection (X : C) :
    GaloisConnection (Sieve.functorPushforward F : Sieve X → Sieve (F.obj X))
      (Sieve.functorPullback F) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X : C
    ⊢ GaloisConnection (CategoryTheory.Sieve.functorPushforward F) (CategoryTheory …
  -/
  intro R S
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X : C
    R : CategoryTheory.Sieve X
    S : CategoryTheory.Sieve (F.obj X)
    ⊢ Iff (LE.le (CategoryTheory.Sieve.functorPushforward F R) S) (LE.le R (Catego …
  -/
  constructor
    /-
      case mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X : C
      R : CategoryTheory.Sieve X
      S : CategoryTheory.Sieve (F.obj X)
      ⊢ LE.le (CategoryTheory.Sieve.functorPushforward F R) S → LE.le R (CategoryThe …
    -/
  · intro hle X f hf
    /-
      case mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X✝ : C
      R : CategoryTheory.Sieve X✝
      S : CategoryTheory.Sieve (F.obj X✝)
      hle : LE.le (CategoryTheory.Sieve.functorPushforward F R) S
      X : C
      f : Quiver.Hom X X✝
      hf : R.arrows f
      ⊢ (CategoryTheory.Sieve.functorPullback F S).arrows f
    -/
    apply hle
    /-
      case mp.a
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X✝ : C
      R : CategoryTheory.Sieve X✝
      S : CategoryTheory.Sieve (F.obj X✝)
      hle : LE.le (CategoryTheory.Sieve.functorPushforward F R) S
      X : C
      f : Quiver.Hom X X✝
      hf : R.arrows f
      ⊢ (CategoryTheory.Sieve.functorPushforward F R).arrows (F.map f)
    -/
    refine ⟨X, f, 𝟙 _, hf, ?_⟩
    /-
      case mp.a
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X✝ : C
      R : CategoryTheory.Sieve X✝
      S : CategoryTheory.Sieve (F.obj X✝)
      hle : LE.le (CategoryTheory.Sieve.functorPushforward F R) S
      X : C
      f : Quiver.Hom X X✝
      hf : R.arrows f
      ⊢ Eq (F.map f) (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStr …
    -/
    rw [id_comp]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X : C
      R : CategoryTheory.Sieve X
      S : CategoryTheory.Sieve (F.obj X)
      ⊢ LE.le R (CategoryTheory.Sieve.functorPullback F S) → LE.le (CategoryTheory.S …
    -/
  · rintro hle Y f ⟨X, g, h, hg, rfl⟩
    /-
      case mpr.intro.intro.intro.intro
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X✝ : C
      R : CategoryTheory.Sieve X✝
      S : CategoryTheory.Sieve (F.obj X✝)
      hle : LE.le R (CategoryTheory.Sieve.functorPullback F S)
      Y : D
      X : C
      g : Quiver.Hom X X✝
      h : Quiver.Hom Y (F.obj X)
      hg : R.arrows g
      ⊢ S.arrows (CategoryTheory.CategoryStruct.comp h (F.map g))
    -/
    apply Sieve.downward_closed S
    /-
      case mpr.intro.intro.intro.intro.x
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X✝ : C
      R : CategoryTheory.Sieve X✝
      S : CategoryTheory.Sieve (F.obj X✝)
      hle : LE.le R (CategoryTheory.Sieve.functorPullback F S)
      Y : D
      X : C
      g : Quiver.Hom X X✝
      h : Quiver.Hom Y (F.obj X)
      hg : R.arrows g
      ⊢ S.arrows (F.map g)
    -/
    exact hle g hg
    /-
      🎉 no goals
    -/


theorem functorPullback_monotone (X : C) :
    Monotone (Sieve.functorPullback F : Sieve (F.obj X) → Sieve X) :=
  (functor_galoisConnection F X).monotone_u


theorem functorPushforward_monotone (X : C) :
    Monotone (Sieve.functorPushforward F : Sieve X → Sieve (F.obj X)) :=
  (functor_galoisConnection F X).monotone_l


theorem le_functorPushforward_pullback (R : Sieve X) :
    R ≤ (R.functorPushforward F).functorPullback F :=
  (functor_galoisConnection F X).le_u_l _


theorem functorPullback_pushforward_le (R : Sieve (F.obj X)) :
    (R.functorPullback F).functorPushforward F ≤ R :=
  (functor_galoisConnection F X).l_u_le _


theorem functorPushforward_union (S R : Sieve X) :
    (S ⊔ R).functorPushforward F = S.functorPushforward F ⊔ R.functorPushforward F :=
  (functor_galoisConnection F X).l_sup


theorem functorPullback_union (S R : Sieve (F.obj X)) :
    (S ⊔ R).functorPullback F = S.functorPullback F ⊔ R.functorPullback F :=
  rfl


theorem functorPullback_inter (S R : Sieve (F.obj X)) :
    (S ⊓ R).functorPullback F = S.functorPullback F ⊓ R.functorPullback F :=
  rfl


@[simp]
theorem functorPushforward_bot (F : C ⥤ D) (X : C) : (⊥ : Sieve X).functorPushforward F = ⊥ :=
  (functor_galoisConnection F X).l_bot


@[simp]
theorem functorPushforward_top (F : C ⥤ D) (X : C) : (⊤ : Sieve X).functorPushforward F = ⊤ := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X : C
    ⊢ Eq (CategoryTheory.Sieve.functorPushforward F Top.top) Top.top
  -/
  refine (generate_sieve _).symm.trans ?_
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X : C
    ⊢ Eq (CategoryTheory.Sieve.generate (CategoryTheory.Sieve.functorPushforward F …
  -/
  apply generate_of_contains_isSplitEpi (𝟙 (F.obj X))
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X : C
    ⊢ (CategoryTheory.Sieve.functorPushforward F Top.top).arrows (CategoryTheory.C …
  -/
  exact ⟨X, 𝟙 _, 𝟙 _, trivial, by simp⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem functorPullback_bot (F : C ⥤ D) (X : C) : (⊥ : Sieve (F.obj X)).functorPullback F = ⊥ :=
  rfl


@[simp]
theorem functorPullback_top (F : C ⥤ D) (X : C) : (⊤ : Sieve (F.obj X)).functorPullback F = ⊤ :=
  rfl


theorem image_mem_functorPushforward (R : Sieve X) {V} {f : V ⟶ X} (h : R f) :
    R.functorPushforward F (F.map f) :=
                    /-
                      C : Type u₁
                      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                      D : Type u₂
                      inst✝ : CategoryTheory.Category.{v₂, u₂} D
                      F : CategoryTheory.Functor C D
                      X : C
                      R : CategoryTheory.Sieve X
                      V : C
                      f : Quiver.Hom V X
                      h : R.arrows f
                      ⊢ Eq (F.map f) (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStr …
                    -/
  ⟨V, f, 𝟙 _, h, by simp⟩
                    /-
                      🎉 no goals
                    -/


/-- When `F` is essentially surjective and full, the galois connection is a galois insertion. -/
def essSurjFullFunctorGaloisInsertion [F.EssSurj] [F.Full] (X : C) :
    GaloisInsertion (Sieve.functorPushforward F : Sieve X → Sieve (F.obj X))
      (Sieve.functorPullback F) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X✝ Y Z : C
    f : Quiver.Hom Y X✝
    S R : CategoryTheory.Sieve X✝
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    inst✝¹ : F.EssSurj
    inst✝ : F.Full
    X : C
    ⊢ GaloisInsertion (CategoryTheory.Sieve.functorPushforward F) (CategoryTheory. …
  -/
  apply (functor_galoisConnection F X).toGaloisInsertion
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X✝ Y Z : C
    f : Quiver.Hom Y X✝
    S R : CategoryTheory.Sieve X✝
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    inst✝¹ : F.EssSurj
    inst✝ : F.Full
    X : C
    ⊢ ∀ (b : CategoryTheory.Sieve (F.obj X)), LE.le b (CategoryTheory.Sieve.functo …
  -/
  intro S Y f hf
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X✝ Y✝ Z : C
    f✝ : Quiver.Hom Y✝ X✝
    S✝ R : CategoryTheory.Sieve X✝
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    inst✝¹ : F.EssSurj
    inst✝ : F.Full
    X : C
    S : CategoryTheory.Sieve (F.obj X)
    Y : D
    f : Quiver.Hom Y (F.obj X)
    hf : S.arrows f
    ⊢ (CategoryTheory.Sieve.functorPushforward F (CategoryTheory.Sieve.functorPull …
  -/
  refine ⟨_, F.preimage ((F.objObjPreimageIso Y).hom ≫ f), (F.objObjPreimageIso Y).inv, ?_⟩
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X✝ Y✝ Z : C
    f✝ : Quiver.Hom Y✝ X✝
    S✝ R : CategoryTheory.Sieve X✝
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    inst✝¹ : F.EssSurj
    inst✝ : F.Full
    X : C
    S : CategoryTheory.Sieve (F.obj X)
    Y : D
    f : Quiver.Hom Y (F.obj X)
    hf : S.arrows f
    ⊢ And ((CategoryTheory.Sieve.functorPullback F S).arrows (F.preimage (Category …
  -/
  simpa using hf
  /-
    🎉 no goals
  -/


/-- When `F` is fully faithful, the galois connection is a galois coinsertion. -/
def fullyFaithfulFunctorGaloisCoinsertion [F.Full] [F.Faithful] (X : C) :
    GaloisCoinsertion (Sieve.functorPushforward F : Sieve X → Sieve (F.obj X))
      (Sieve.functorPullback F) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X✝ Y Z : C
    f : Quiver.Hom Y X✝
    S R : CategoryTheory.Sieve X✝
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    X : C
    ⊢ GaloisCoinsertion (CategoryTheory.Sieve.functorPushforward F) (CategoryTheor …
  -/
  apply (functor_galoisConnection F X).toGaloisCoinsertion
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X✝ Y Z : C
    f : Quiver.Hom Y X✝
    S R : CategoryTheory.Sieve X✝
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    X : C
    ⊢ ∀ (a : CategoryTheory.Sieve X), LE.le (CategoryTheory.Sieve.functorPullback  …
  -/
  rintro S Y f ⟨Z, g, h, h₁, h₂⟩
  /-
    case intro.intro.intro.intro
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X✝ Y✝ Z✝ : C
    f✝ : Quiver.Hom Y✝ X✝
    S✝ R : CategoryTheory.Sieve X✝
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    X : C
    S : CategoryTheory.Sieve X
    Y : C
    f : Quiver.Hom Y X
    Z : C
    g : Quiver.Hom Z X
    h : Quiver.Hom (F.obj Y) (F.obj Z)
    h₁ : S.arrows g
    h₂ : Eq (F.map f) (CategoryTheory.CategoryStruct.comp h (F.map g))
    ⊢ S.arrows f
  -/
  rw [← F.map_preimage h, ← F.map_comp] at h₂
  /-
    case intro.intro.intro.intro
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X✝ Y✝ Z✝ : C
    f✝ : Quiver.Hom Y✝ X✝
    S✝ R : CategoryTheory.Sieve X✝
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    X : C
    S : CategoryTheory.Sieve X
    Y : C
    f : Quiver.Hom Y X
    Z : C
    g : Quiver.Hom Z X
    h : Quiver.Hom (F.obj Y) (F.obj Z)
    h₁ : S.arrows g
    h₂ : Eq (F.map f) (F.map (CategoryTheory.CategoryStruct.comp (F.preimage h) g))
    ⊢ S.arrows f
  -/
  rw [F.map_injective h₂]
  /-
    case intro.intro.intro.intro
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X✝ Y✝ Z✝ : C
    f✝ : Quiver.Hom Y✝ X✝
    S✝ R : CategoryTheory.Sieve X✝
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    X : C
    S : CategoryTheory.Sieve X
    Y : C
    f : Quiver.Hom Y X
    Z : C
    g : Quiver.Hom Z X
    h : Quiver.Hom (F.obj Y) (F.obj Z)
    h₁ : S.arrows g
    h₂ : Eq (F.map f) (F.map (CategoryTheory.CategoryStruct.comp (F.preimage h) g))
    ⊢ S.arrows (CategoryTheory.CategoryStruct.comp (F.preimage h) g)
  -/
  exact S.downward_closed h₁ _
  /-
    🎉 no goals
  -/


lemma functorPushforward_functor (S : Sieve X) (e : C ≌ D) :
    S.functorPushforward e.functor = (S.pullback (e.unitInv.app X)).functorPullback e.inverse := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    X : C
    S : CategoryTheory.Sieve X
    e : CategoryTheory.Equivalence C D
    ⊢ Eq (CategoryTheory.Sieve.functorPushforward e.functor S) (CategoryTheory.Sie …
  -/
  ext Y iYX
  /-
    case h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    X : C
    S : CategoryTheory.Sieve X
    e : CategoryTheory.Equivalence C D
    Y : D
    iYX : Quiver.Hom Y (e.functor.obj X)
    ⊢ Iff ((CategoryTheory.Sieve.functorPushforward e.functor S).arrows iYX) ((Cat …
  -/
  constructor
    /-
      case h.mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      X : C
      S : CategoryTheory.Sieve X
      e : CategoryTheory.Equivalence C D
      Y : D
      iYX : Quiver.Hom Y (e.functor.obj X)
      ⊢ (CategoryTheory.Sieve.functorPushforward e.functor S).arrows iYX → (Category …
    -/
  · rintro ⟨Z, iZX, iYZ, hiZX, rfl⟩
    /-
      case h.mp.intro.intro.intro.intro
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      X : C
      S : CategoryTheory.Sieve X
      e : CategoryTheory.Equivalence C D
      Y : D
      Z : C
      iZX : Quiver.Hom Z X
      iYZ : Quiver.Hom Y (e.functor.obj Z)
      hiZX : S.arrows iZX
      ⊢ (CategoryTheory.Sieve.functorPullback e.inverse (CategoryTheory.Sieve.pullba …
    -/
    simpa using S.downward_closed hiZX (e.inverse.map iYZ ≫ e.unitInv.app Z)
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      X : C
      S : CategoryTheory.Sieve X
      e : CategoryTheory.Equivalence C D
      Y : D
      iYX : Quiver.Hom Y (e.functor.obj X)
      ⊢ (CategoryTheory.Sieve.functorPullback e.inverse (CategoryTheory.Sieve.pullba …
    -/
  · intro H
    /-
      case h.mpr
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      X : C
      S : CategoryTheory.Sieve X
      e : CategoryTheory.Equivalence C D
      Y : D
      iYX : Quiver.Hom Y (e.functor.obj X)
      H : (CategoryTheory.Sieve.functorPullback e.inverse (CategoryTheory.Sieve.pull …
      ⊢ (CategoryTheory.Sieve.functorPushforward e.functor S).arrows iYX
    -/
    exact ⟨_, e.inverse.map iYX ≫ e.unitInv.app X, e.counitInv.app Y, by simpa using H, by simp⟩
    /-
      🎉 no goals
    -/


@[simp]
lemma mem_functorPushforward_functor {Y : D} {S : Sieve X} {e : C ≌ D} {f : Y ⟶ e.functor.obj X} :
    S.functorPushforward e.functor f ↔ S (e.inverse.map f ≫ e.unitInv.app X) :=
  congr($(S.functorPushforward_functor e).arrows f)


lemma functorPushforward_inverse {X : D} (S : Sieve X) (e : C ≌ D) :
    S.functorPushforward e.inverse = (S.pullback (e.counit.app X)).functorPullback e.functor :=
  Sieve.functorPushforward_functor S e.symm


@[simp]
lemma mem_functorPushforward_inverse {X : D} {S : Sieve X} {e : C ≌ D} {f : Y ⟶ e.inverse.obj X} :
    S.functorPushforward e.inverse f ↔ S (e.functor.map f ≫ e.counit.app X) :=
  congr($(S.functorPushforward_inverse e).arrows f)


lemma functorPushforward_equivalence_eq_pullback {U : C} (S : Sieve U) :
    Sieve.functorPushforward e.inverse (Sieve.functorPushforward e.functor S) =
                                               /-
                                                 C : Type u₁
                                                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                 D : Type u₂
                                                 inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                 e : CategoryTheory.Equivalence C D
                                                 U : C
                                                 S : CategoryTheory.Sieve U
                                                 ⊢ Eq (CategoryTheory.Sieve.functorPushforward e.inverse (CategoryTheory.Sieve. …
                                               -/
      Sieve.pullback (e.unitInv.app U) S := by ext; simp
                                                    /-
                                                      🎉 no goals
                                                    -/


lemma pullback_functorPushforward_equivalence_eq {X : C} (S : Sieve X) :
    Sieve.pullback (e.unit.app X) (Sieve.functorPushforward e.inverse
                                                        /-
                                                          C : Type u₁
                                                          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                          D : Type u₂
                                                          inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                          e : CategoryTheory.Equivalence C D
                                                          X : C
                                                          S : CategoryTheory.Sieve X
                                                          ⊢ Eq (CategoryTheory.Sieve.pullback (e.unit.app X) (CategoryTheory.Sieve.funct …
                                                        -/
      (Sieve.functorPushforward e.functor S)) = S := by ext; simp
                                                             /-
                                                               🎉 no goals
                                                             -/


lemma mem_functorPushforward_iff_of_full [F.Full] {X Y : C} (R : Sieve X) (f : F.obj Y ⟶ F.obj X) :
    (R.arrows.functorPushforward F) f ↔ ∃ (g : Y ⟶ X), F.map g = f ∧ R g := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : F.Full
    X Y : C
    R : CategoryTheory.Sieve X
    f : Quiver.Hom (F.obj Y) (F.obj X)
    ⊢ Iff (CategoryTheory.Presieve.functorPushforward F R.arrows f) (Exists fun g  …
  -/
  refine ⟨fun ⟨Z, g, h, hg, hcomp⟩ ↦ ?_, fun ⟨g, hcomp, hg⟩ ↦ ?_⟩
    /-
      case refine_1
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : F.Full
      X Y : C
      R : CategoryTheory.Sieve X
      f : Quiver.Hom (F.obj Y) (F.obj X)
      x✝ : CategoryTheory.Presieve.functorPushforward F R.arrows f
      Z : C
      g : Quiver.Hom Z X
      h : Quiver.Hom (F.obj Y) (F.obj Z)
      hg : R.arrows g
      hcomp : Eq f (CategoryTheory.CategoryStruct.comp h (F.map g))
      ⊢ Exists fun g => And (Eq (F.map g) f) (R.arrows g)
    -/
  · obtain ⟨h', hh'⟩ := F.map_surjective h
    /-
      case refine_1.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : F.Full
      X Y : C
      R : CategoryTheory.Sieve X
      f : Quiver.Hom (F.obj Y) (F.obj X)
      x✝ : CategoryTheory.Presieve.functorPushforward F R.arrows f
      Z : C
      g : Quiver.Hom Z X
      h : Quiver.Hom (F.obj Y) (F.obj Z)
      hg : R.arrows g
      hcomp : Eq f (CategoryTheory.CategoryStruct.comp h (F.map g))
      h' : Quiver.Hom Y Z
      hh' : Eq (F.map h') h
      ⊢ Exists fun g => And (Eq (F.map g) f) (R.arrows g)
    -/
    use h' ≫ g
    /-
      case h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : F.Full
      X Y : C
      R : CategoryTheory.Sieve X
      f : Quiver.Hom (F.obj Y) (F.obj X)
      x✝ : CategoryTheory.Presieve.functorPushforward F R.arrows f
      Z : C
      g : Quiver.Hom Z X
      h : Quiver.Hom (F.obj Y) (F.obj Z)
      hg : R.arrows g
      hcomp : Eq f (CategoryTheory.CategoryStruct.comp h (F.map g))
      h' : Quiver.Hom Y Z
      hh' : Eq (F.map h') h
      ⊢ And (Eq (F.map (CategoryTheory.CategoryStruct.comp h' g)) f) (R.arrows (Cate …
    -/
    simp only [Functor.map_comp, hh', hcomp, true_and]
    /-
      case h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : F.Full
      X Y : C
      R : CategoryTheory.Sieve X
      f : Quiver.Hom (F.obj Y) (F.obj X)
      x✝ : CategoryTheory.Presieve.functorPushforward F R.arrows f
      Z : C
      g : Quiver.Hom Z X
      h : Quiver.Hom (F.obj Y) (F.obj Z)
      hg : R.arrows g
      hcomp : Eq f (CategoryTheory.CategoryStruct.comp h (F.map g))
      h' : Quiver.Hom Y Z
      hh' : Eq (F.map h') h
      ⊢ R.arrows (CategoryTheory.CategoryStruct.comp h' g)
    -/
    apply R.downward_closed hg
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : F.Full
      X Y : C
      R : CategoryTheory.Sieve X
      f : Quiver.Hom (F.obj Y) (F.obj X)
      x✝ : Exists fun g => And (Eq (F.map g) f) (R.arrows g)
      g : Quiver.Hom Y X
      hcomp : Eq (F.map g) f
      hg : R.arrows g
      ⊢ CategoryTheory.Presieve.functorPushforward F R.arrows f
    -/
  · use Y, g, 𝟙 _, hg
    /-
      case right
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : F.Full
      X Y : C
      R : CategoryTheory.Sieve X
      f : Quiver.Hom (F.obj Y) (F.obj X)
      x✝ : Exists fun g => And (Eq (F.map g) f) (R.arrows g)
      g : Quiver.Hom Y X
      hcomp : Eq (F.map g) f
      hg : R.arrows g
      ⊢ Eq f (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ( …
    -/
    simp [hcomp]
    /-
      🎉 no goals
    -/


lemma mem_functorPushforward_iff_of_full_of_faithful [F.Full] [F.Faithful]
    {X Y : C} (R : Sieve X) (f : Y ⟶ X) :
    (R.arrows.functorPushforward F) (F.map f) ↔ R f := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    X Y : C
    R : CategoryTheory.Sieve X
    f : Quiver.Hom Y X
    ⊢ Iff (CategoryTheory.Presieve.functorPushforward F R.arrows (F.map f)) (R.arr …
  -/
  rw [Sieve.mem_functorPushforward_iff_of_full]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    X Y : C
    R : CategoryTheory.Sieve X
    f : Quiver.Hom Y X
    ⊢ Iff (Exists fun g => And (Eq (F.map g) (F.map f)) (R.arrows g)) (R.arrows f)
  -/
  refine ⟨fun ⟨g, hcomp, hg⟩ ↦ ?_, fun hf ↦ ⟨f, rfl, hf⟩⟩
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    X Y : C
    R : CategoryTheory.Sieve X
    f : Quiver.Hom Y X
    x✝ : Exists fun g => And (Eq (F.map g) (F.map f)) (R.arrows g)
    g : Quiver.Hom Y X
    hcomp : Eq (F.map g) (F.map f)
    hg : R.arrows g
    ⊢ R.arrows f
  -/
  rwa [← F.map_injective hcomp]
  /-
    🎉 no goals
  -/


/-- A sieve induces a presheaf. -/
@[simps]
def functor (S : Sieve X) : Cᵒᵖ ⥤ Type v₁ where
  obj Y := { g : Y.unop ⟶ X // S g }
  map f g := ⟨f.unop ≫ g.1, downward_closed _ g.2 _⟩


/-- If a sieve S is contained in a sieve T, then we have a morphism of presheaves on their induced
presheaves.
-/
@[simps]
def natTransOfLe {S T : Sieve X} (h : S ≤ T) : S.functor ⟶ T.functor where app _ f := ⟨f.1, h _ f.2⟩


/-- The natural inclusion from the functor induced by a sieve to the yoneda embedding. -/
@[simps]
def functorInclusion (S : Sieve X) : S.functor ⟶ yoneda.obj X where app _ f := f.1


theorem natTransOfLe_comm {S T : Sieve X} (h : S ≤ T) :
    natTransOfLe h ≫ functorInclusion _ = functorInclusion _ :=
  rfl


/-- The presheaf induced by a sieve is a subobject of the yoneda embedding. -/
instance functorInclusion_is_mono : Mono S.functorInclusion :=
  ⟨fun f g h => by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y Z : C
      f✝ : Quiver.Hom Y X
      S R : CategoryTheory.Sieve X
      Z✝ : CategoryTheory.Functor (Opposite C) (Type v₁)
      f g : Quiver.Hom Z✝ S.functor
      h : Eq (CategoryTheory.CategoryStruct.comp f S.functorInclusion) (CategoryTheo …
      ⊢ Eq f g
    -/
    ext Y y
    /-
      case w.h.h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y✝ Z : C
      f✝ : Quiver.Hom Y✝ X
      S R : CategoryTheory.Sieve X
      Z✝ : CategoryTheory.Functor (Opposite C) (Type v₁)
      f g : Quiver.Hom Z✝ S.functor
      h : Eq (CategoryTheory.CategoryStruct.comp f S.functorInclusion) (CategoryTheo …
      Y : Opposite C
      y : Z✝.obj Y
      ⊢ Eq (f.app Y y) (g.app Y y)
    -/
    simpa [Subtype.ext_iff_val] using congr_fun (NatTrans.congr_app h Y) y⟩
    /-
      🎉 no goals
    -/

-- TODO: Show that when `f` is mono, this is right inverse to `functorInclusion` up to isomorphism.

/-- A natural transformation to a representable functor induces a sieve. This is the left inverse of
`functorInclusion`, shown in `sieveOfSubfunctor_functorInclusion`.
-/
@[simps]
def sieveOfSubfunctor {R} (f : R ⟶ yoneda.obj X) : Sieve X where
  arrows Y g := ∃ t, f.app (Opposite.op Y) t = g
  downward_closed := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y Z : C
      f✝ : Quiver.Hom Y X
      S R✝ : CategoryTheory.Sieve X
      R : CategoryTheory.Functor (Opposite C) (Type v₁)
      f : Quiver.Hom R (CategoryTheory.yoneda.obj X)
      ⊢ ∀ {Y Z : C} {f_1 : Quiver.Hom Y X}, (fun Y g => Exists fun t => Eq (f.app {  …
    -/
    rintro Y Z _ ⟨t, rfl⟩ g
    /-
      case intro
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y✝ Z✝ : C
      f✝ : Quiver.Hom Y✝ X
      S R✝ : CategoryTheory.Sieve X
      R : CategoryTheory.Functor (Opposite C) (Type v₁)
      f : Quiver.Hom R (CategoryTheory.yoneda.obj X)
      Y Z : C
      t : R.obj { unop := Y }
      g : Quiver.Hom Z Y
      ⊢ Exists fun t_1 => Eq (f.app { unop := Z } t_1) (CategoryTheory.CategoryStruc …
    -/
    refine ⟨R.map g.op t, ?_⟩
    /-
      case intro
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y✝ Z✝ : C
      f✝ : Quiver.Hom Y✝ X
      S R✝ : CategoryTheory.Sieve X
      R : CategoryTheory.Functor (Opposite C) (Type v₁)
      f : Quiver.Hom R (CategoryTheory.yoneda.obj X)
      Y Z : C
      t : R.obj { unop := Y }
      g : Quiver.Hom Z Y
      ⊢ Eq (f.app { unop := Z } (R.map g.op t)) (CategoryTheory.CategoryStruct.comp  …
    -/
    rw [FunctorToTypes.naturality _ _ f]
    /-
      case intro
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y✝ Z✝ : C
      f✝ : Quiver.Hom Y✝ X
      S R✝ : CategoryTheory.Sieve X
      R : CategoryTheory.Functor (Opposite C) (Type v₁)
      f : Quiver.Hom R (CategoryTheory.yoneda.obj X)
      Y Z : C
      t : R.obj { unop := Y }
      g : Quiver.Hom Z Y
      ⊢ Eq ((CategoryTheory.yoneda.obj X).map g.op (f.app { unop := Y } t)) (Categor …
    -/
    simp
    /-
      🎉 no goals
    -/


theorem sieveOfSubfunctor_functorInclusion : sieveOfSubfunctor S.functorInclusion = S := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    S : CategoryTheory.Sieve X
    ⊢ Eq (CategoryTheory.Sieve.sieveOfSubfunctor S.functorInclusion) S
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    S : CategoryTheory.Sieve X
    Y✝ : C
    f✝ : Quiver.Hom Y✝ X
    ⊢ Iff ((CategoryTheory.Sieve.sieveOfSubfunctor S.functorInclusion).arrows f✝)  …
  -/
  simp only [functorInclusion_app, sieveOfSubfunctor_apply]
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    S : CategoryTheory.Sieve X
    Y✝ : C
    f✝ : Quiver.Hom Y✝ X
    ⊢ Iff (Exists fun t => Eq (↑t) f✝) (S.arrows f✝)
  -/
  constructor
    /-
      case h.mp
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      S : CategoryTheory.Sieve X
      Y✝ : C
      f✝ : Quiver.Hom Y✝ X
      ⊢ (Exists fun t => Eq (↑t) f✝) → S.arrows f✝
    -/
  · rintro ⟨⟨f, hf⟩, rfl⟩
    /-
      case h.mp.intro.mk
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      S : CategoryTheory.Sieve X
      Y✝ : C
      f : Quiver.Hom (Opposite.unop { unop := Y✝ }) X
      hf : S.arrows f
      ⊢ S.arrows ↑⟨f, hf⟩
    -/
    exact hf
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      S : CategoryTheory.Sieve X
      Y✝ : C
      f✝ : Quiver.Hom Y✝ X
      ⊢ S.arrows f✝ → Exists fun t => Eq (↑t) f✝
    -/
  · intro hf
    /-
      case h.mpr
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      S : CategoryTheory.Sieve X
      Y✝ : C
      f✝ : Quiver.Hom Y✝ X
      hf : S.arrows f✝
      ⊢ Exists fun t => Eq (↑t) f✝
    -/
    exact ⟨⟨_, hf⟩, rfl⟩
    /-
      🎉 no goals
    -/


instance functorInclusion_top_isIso : IsIso (⊤ : Sieve X).functorInclusion :=
  ⟨⟨{ app := fun _ a => ⟨a, ⟨⟩⟩ }, rfl, rfl⟩⟩


