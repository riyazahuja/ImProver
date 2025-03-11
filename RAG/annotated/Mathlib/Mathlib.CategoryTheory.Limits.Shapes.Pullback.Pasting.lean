/-- The `PullbackCone` obtained by pasting two `PullbackCone`'s horizontally -/
abbrev PullbackCone.pasteHoriz
    (t₂ : PullbackCone g₂ i₃) {i₂ : t₂.pt ⟶ Y₂} (t₁ : PullbackCone g₁ i₂) (hi₂ : i₂ = t₂.fst) :
    PullbackCone (g₁ ≫ g₂) i₃ :=
  PullbackCone.mk t₁.fst (t₁.snd ≫ t₂.snd)
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X₃ Y₁ Y₂ Y₃ : C
          g₁ : Quiver.Hom Y₁ Y₂
          g₂ : Quiver.Hom Y₂ Y₃
          i₃ : Quiver.Hom X₃ Y₃
          t₂ : CategoryTheory.Limits.PullbackCone g₂ i₃
          i₂ : Quiver.Hom t₂.pt Y₂
          t₁ : CategoryTheory.Limits.PullbackCone g₁ i₂
          hi₂ : Eq i₂ t₂.fst
          ⊢ Eq (CategoryTheory.CategoryStruct.comp t₁.fst (CategoryTheory.CategoryStruct …
        -/
    (by rw [reassoc_of% t₁.condition, Category.assoc, ← t₂.condition, ← hi₂])
        /-
          🎉 no goals
        -/


local notation "f₂" => t₂.snd

local notation "X₁" => t₁.pt

local notation "i₁" => t₁.fst

local notation "f₁" => t₁.snd


/-- Given
```
X₁ - f₁ -> X₂ - f₂ -> X₃
|          |          |
i₁         i₂         i₃
↓          ↓          ↓
Y₁ - g₁ -> Y₂ - g₂ -> Y₃
```
Then the big square is a pullback if both the small squares are.
-/
def pasteHorizIsPullback (H : IsLimit t₂) (H' : IsLimit t₁) : IsLimit (t₂.pasteHoriz t₁ hi₂) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X₃ Y₁ Y₂ Y₃ : C
    g₁ : Quiver.Hom Y₁ Y₂
    g₂ : Quiver.Hom Y₂ Y₃
    i₃ : Quiver.Hom X₃ Y₃
    t₂ : CategoryTheory.Limits.PullbackCone g₂ i₃
    i₂ : Quiver.Hom t₂.pt Y₂
    t₁ : CategoryTheory.Limits.PullbackCone g₁ i₂
    hi₂ : Eq i₂ t₂.fst
    H : CategoryTheory.Limits.IsLimit t₂
    H' : CategoryTheory.Limits.IsLimit t₁
    ⊢ CategoryTheory.Limits.IsLimit (t₂.pasteHoriz t₁ hi₂)
  -/
  apply PullbackCone.isLimitAux'
  /-
    case create
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X₃ Y₁ Y₂ Y₃ : C
    g₁ : Quiver.Hom Y₁ Y₂
    g₂ : Quiver.Hom Y₂ Y₃
    i₃ : Quiver.Hom X₃ Y₃
    t₂ : CategoryTheory.Limits.PullbackCone g₂ i₃
    i₂ : Quiver.Hom t₂.pt Y₂
    t₁ : CategoryTheory.Limits.PullbackCone g₁ i₂
    hi₂ : Eq i₂ t₂.fst
    H : CategoryTheory.Limits.IsLimit t₂
    H' : CategoryTheory.Limits.IsLimit t₁
    ⊢ (s : CategoryTheory.Limits.PullbackCone (CategoryTheory.CategoryStruct.comp  …
  -/
  intro s
  -- Obtain the lift from lifting from both the small squares consecutively.
  obtain ⟨l₂, hl₂, hl₂'⟩ := PullbackCone.IsLimit.lift' H (s.fst ≫ g₁) s.snd
    (by rw [← s.condition, Category.assoc])
  /-
    case create.mk.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X₃ Y₁ Y₂ Y₃ : C
    g₁ : Quiver.Hom Y₁ Y₂
    g₂ : Quiver.Hom Y₂ Y₃
    i₃ : Quiver.Hom X₃ Y₃
    t₂ : CategoryTheory.Limits.PullbackCone g₂ i₃
    i₂ : Quiver.Hom t₂.pt Y₂
    t₁ : CategoryTheory.Limits.PullbackCone g₁ i₂
    hi₂ : Eq i₂ t₂.fst
    H : CategoryTheory.Limits.IsLimit t₂
    H' : CategoryTheory.Limits.IsLimit t₁
    s : CategoryTheory.Limits.PullbackCone (CategoryTheory.CategoryStruct.comp g₁  …
    l₂ : Quiver.Hom s.pt t₂.pt
    hl₂ : Eq (CategoryTheory.CategoryStruct.comp l₂ t₂.fst) (CategoryTheory.Catego …
    hl₂' : Eq (CategoryTheory.CategoryStruct.comp l₂ t₂.snd) s.snd
    ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp l (t₂.pasteHori …
  -/
  obtain ⟨l₁, hl₁, hl₁'⟩ := PullbackCone.IsLimit.lift' H' s.fst l₂ (by rw [← hl₂, hi₂])
  /-
    case create.mk.intro.mk.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X₃ Y₁ Y₂ Y₃ : C
    g₁ : Quiver.Hom Y₁ Y₂
    g₂ : Quiver.Hom Y₂ Y₃
    i₃ : Quiver.Hom X₃ Y₃
    t₂ : CategoryTheory.Limits.PullbackCone g₂ i₃
    i₂ : Quiver.Hom t₂.pt Y₂
    t₁ : CategoryTheory.Limits.PullbackCone g₁ i₂
    hi₂ : Eq i₂ t₂.fst
    H : CategoryTheory.Limits.IsLimit t₂
    H' : CategoryTheory.Limits.IsLimit t₁
    s : CategoryTheory.Limits.PullbackCone (CategoryTheory.CategoryStruct.comp g₁  …
    l₂ : Quiver.Hom s.pt t₂.pt
    hl₂ : Eq (CategoryTheory.CategoryStruct.comp l₂ t₂.fst) (CategoryTheory.Catego …
    hl₂' : Eq (CategoryTheory.CategoryStruct.comp l₂ t₂.snd) s.snd
    l₁ : Quiver.Hom s.pt t₁.pt
    hl₁ : Eq (CategoryTheory.CategoryStruct.comp l₁ t₁.fst) s.fst
    hl₁' : Eq (CategoryTheory.CategoryStruct.comp l₁ t₁.snd) l₂
    ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp l (t₂.pasteHori …
  -/
  refine ⟨l₁, hl₁, by simp [reassoc_of% hl₁', hl₂'], ?_⟩
  -- Uniqueness also follows from the universal property of both the small squares.
  /-
    case create.mk.intro.mk.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X₃ Y₁ Y₂ Y₃ : C
    g₁ : Quiver.Hom Y₁ Y₂
    g₂ : Quiver.Hom Y₂ Y₃
    i₃ : Quiver.Hom X₃ Y₃
    t₂ : CategoryTheory.Limits.PullbackCone g₂ i₃
    i₂ : Quiver.Hom t₂.pt Y₂
    t₁ : CategoryTheory.Limits.PullbackCone g₁ i₂
    hi₂ : Eq i₂ t₂.fst
    H : CategoryTheory.Limits.IsLimit t₂
    H' : CategoryTheory.Limits.IsLimit t₁
    s : CategoryTheory.Limits.PullbackCone (CategoryTheory.CategoryStruct.comp g₁  …
    l₂ : Quiver.Hom s.pt t₂.pt
    hl₂ : Eq (CategoryTheory.CategoryStruct.comp l₂ t₂.fst) (CategoryTheory.Catego …
    hl₂' : Eq (CategoryTheory.CategoryStruct.comp l₂ t₂.snd) s.snd
    l₁ : Quiver.Hom s.pt t₁.pt
    hl₁ : Eq (CategoryTheory.CategoryStruct.comp l₁ t₁.fst) s.fst
    hl₁' : Eq (CategoryTheory.CategoryStruct.comp l₁ t₁.snd) l₂
    ⊢ ∀ {m : Quiver.Hom s.pt (t₂.pasteHoriz t₁ hi₂).pt}, Eq (CategoryTheory.Catego …
  -/
  intro m hm₁ hm₂
  /-
    case create.mk.intro.mk.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X₃ Y₁ Y₂ Y₃ : C
    g₁ : Quiver.Hom Y₁ Y₂
    g₂ : Quiver.Hom Y₂ Y₃
    i₃ : Quiver.Hom X₃ Y₃
    t₂ : CategoryTheory.Limits.PullbackCone g₂ i₃
    i₂ : Quiver.Hom t₂.pt Y₂
    t₁ : CategoryTheory.Limits.PullbackCone g₁ i₂
    hi₂ : Eq i₂ t₂.fst
    H : CategoryTheory.Limits.IsLimit t₂
    H' : CategoryTheory.Limits.IsLimit t₁
    s : CategoryTheory.Limits.PullbackCone (CategoryTheory.CategoryStruct.comp g₁  …
    l₂ : Quiver.Hom s.pt t₂.pt
    hl₂ : Eq (CategoryTheory.CategoryStruct.comp l₂ t₂.fst) (CategoryTheory.Catego …
    hl₂' : Eq (CategoryTheory.CategoryStruct.comp l₂ t₂.snd) s.snd
    l₁ : Quiver.Hom s.pt t₁.pt
    hl₁ : Eq (CategoryTheory.CategoryStruct.comp l₁ t₁.fst) s.fst
    hl₁' : Eq (CategoryTheory.CategoryStruct.comp l₁ t₁.snd) l₂
    m : Quiver.Hom s.pt (t₂.pasteHoriz t₁ hi₂).pt
    hm₁ : Eq (CategoryTheory.CategoryStruct.comp m (t₂.pasteHoriz t₁ hi₂).fst) s.fst
    hm₂ : Eq (CategoryTheory.CategoryStruct.comp m (t₂.pasteHoriz t₁ hi₂).snd) s.snd
    ⊢ Eq m l₁
  -/
  apply PullbackCone.IsLimit.hom_ext H' (by simpa [hl₁] using hm₁)
  /-
    case create.mk.intro.mk.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X₃ Y₁ Y₂ Y₃ : C
    g₁ : Quiver.Hom Y₁ Y₂
    g₂ : Quiver.Hom Y₂ Y₃
    i₃ : Quiver.Hom X₃ Y₃
    t₂ : CategoryTheory.Limits.PullbackCone g₂ i₃
    i₂ : Quiver.Hom t₂.pt Y₂
    t₁ : CategoryTheory.Limits.PullbackCone g₁ i₂
    hi₂ : Eq i₂ t₂.fst
    H : CategoryTheory.Limits.IsLimit t₂
    H' : CategoryTheory.Limits.IsLimit t₁
    s : CategoryTheory.Limits.PullbackCone (CategoryTheory.CategoryStruct.comp g₁  …
    l₂ : Quiver.Hom s.pt t₂.pt
    hl₂ : Eq (CategoryTheory.CategoryStruct.comp l₂ t₂.fst) (CategoryTheory.Catego …
    hl₂' : Eq (CategoryTheory.CategoryStruct.comp l₂ t₂.snd) s.snd
    l₁ : Quiver.Hom s.pt t₁.pt
    hl₁ : Eq (CategoryTheory.CategoryStruct.comp l₁ t₁.fst) s.fst
    hl₁' : Eq (CategoryTheory.CategoryStruct.comp l₁ t₁.snd) l₂
    m : Quiver.Hom s.pt (t₂.pasteHoriz t₁ hi₂).pt
    hm₁ : Eq (CategoryTheory.CategoryStruct.comp m (t₂.pasteHoriz t₁ hi₂).fst) s.fst
    hm₂ : Eq (CategoryTheory.CategoryStruct.comp m (t₂.pasteHoriz t₁ hi₂).snd) s.snd
    ⊢ Eq (CategoryTheory.CategoryStruct.comp m t₁.snd) (CategoryTheory.CategoryStr …
  -/
  apply PullbackCone.IsLimit.hom_ext H
    /-
      case create.mk.intro.mk.intro.h₀
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X₃ Y₁ Y₂ Y₃ : C
      g₁ : Quiver.Hom Y₁ Y₂
      g₂ : Quiver.Hom Y₂ Y₃
      i₃ : Quiver.Hom X₃ Y₃
      t₂ : CategoryTheory.Limits.PullbackCone g₂ i₃
      i₂ : Quiver.Hom t₂.pt Y₂
      t₁ : CategoryTheory.Limits.PullbackCone g₁ i₂
      hi₂ : Eq i₂ t₂.fst
      H : CategoryTheory.Limits.IsLimit t₂
      H' : CategoryTheory.Limits.IsLimit t₁
      s : CategoryTheory.Limits.PullbackCone (CategoryTheory.CategoryStruct.comp g₁  …
      l₂ : Quiver.Hom s.pt t₂.pt
      hl₂ : Eq (CategoryTheory.CategoryStruct.comp l₂ t₂.fst) (CategoryTheory.Catego …
      hl₂' : Eq (CategoryTheory.CategoryStruct.comp l₂ t₂.snd) s.snd
      l₁ : Quiver.Hom s.pt t₁.pt
      hl₁ : Eq (CategoryTheory.CategoryStruct.comp l₁ t₁.fst) s.fst
      hl₁' : Eq (CategoryTheory.CategoryStruct.comp l₁ t₁.snd) l₂
      m : Quiver.Hom s.pt (t₂.pasteHoriz t₁ hi₂).pt
      hm₁ : Eq (CategoryTheory.CategoryStruct.comp m (t₂.pasteHoriz t₁ hi₂).fst) s.fst
      hm₂ : Eq (CategoryTheory.CategoryStruct.comp m (t₂.pasteHoriz t₁ hi₂).snd) s.snd
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp m …
    -/
  · dsimp at hm₁
    /-
      case create.mk.intro.mk.intro.h₀
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X₃ Y₁ Y₂ Y₃ : C
      g₁ : Quiver.Hom Y₁ Y₂
      g₂ : Quiver.Hom Y₂ Y₃
      i₃ : Quiver.Hom X₃ Y₃
      t₂ : CategoryTheory.Limits.PullbackCone g₂ i₃
      i₂ : Quiver.Hom t₂.pt Y₂
      t₁ : CategoryTheory.Limits.PullbackCone g₁ i₂
      hi₂ : Eq i₂ t₂.fst
      H : CategoryTheory.Limits.IsLimit t₂
      H' : CategoryTheory.Limits.IsLimit t₁
      s : CategoryTheory.Limits.PullbackCone (CategoryTheory.CategoryStruct.comp g₁  …
      l₂ : Quiver.Hom s.pt t₂.pt
      hl₂ : Eq (CategoryTheory.CategoryStruct.comp l₂ t₂.fst) (CategoryTheory.Catego …
      hl₂' : Eq (CategoryTheory.CategoryStruct.comp l₂ t₂.snd) s.snd
      l₁ : Quiver.Hom s.pt t₁.pt
      hl₁ : Eq (CategoryTheory.CategoryStruct.comp l₁ t₁.fst) s.fst
      hl₁' : Eq (CategoryTheory.CategoryStruct.comp l₁ t₁.snd) l₂
      m : Quiver.Hom s.pt (t₂.pasteHoriz t₁ hi₂).pt
      hm₁ : Eq (CategoryTheory.CategoryStruct.comp m t₁.fst) s.fst
      hm₂ : Eq (CategoryTheory.CategoryStruct.comp m (t₂.pasteHoriz t₁ hi₂).snd) s.snd
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp m …
    -/
    rw [Category.assoc, ← hi₂, ← t₁.condition, reassoc_of% hm₁, hl₁', hi₂, hl₂]
    /-
      🎉 no goals
    -/
    /-
      case create.mk.intro.mk.intro.h₁
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X₃ Y₁ Y₂ Y₃ : C
      g₁ : Quiver.Hom Y₁ Y₂
      g₂ : Quiver.Hom Y₂ Y₃
      i₃ : Quiver.Hom X₃ Y₃
      t₂ : CategoryTheory.Limits.PullbackCone g₂ i₃
      i₂ : Quiver.Hom t₂.pt Y₂
      t₁ : CategoryTheory.Limits.PullbackCone g₁ i₂
      hi₂ : Eq i₂ t₂.fst
      H : CategoryTheory.Limits.IsLimit t₂
      H' : CategoryTheory.Limits.IsLimit t₁
      s : CategoryTheory.Limits.PullbackCone (CategoryTheory.CategoryStruct.comp g₁  …
      l₂ : Quiver.Hom s.pt t₂.pt
      hl₂ : Eq (CategoryTheory.CategoryStruct.comp l₂ t₂.fst) (CategoryTheory.Catego …
      hl₂' : Eq (CategoryTheory.CategoryStruct.comp l₂ t₂.snd) s.snd
      l₁ : Quiver.Hom s.pt t₁.pt
      hl₁ : Eq (CategoryTheory.CategoryStruct.comp l₁ t₁.fst) s.fst
      hl₁' : Eq (CategoryTheory.CategoryStruct.comp l₁ t₁.snd) l₂
      m : Quiver.Hom s.pt (t₂.pasteHoriz t₁ hi₂).pt
      hm₁ : Eq (CategoryTheory.CategoryStruct.comp m (t₂.pasteHoriz t₁ hi₂).fst) s.fst
      hm₂ : Eq (CategoryTheory.CategoryStruct.comp m (t₂.pasteHoriz t₁ hi₂).snd) s.snd
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp m …
    -/
  · simpa [hl₁', hl₂'] using hm₂
    /-
      🎉 no goals
    -/


/-- Given
```
X₁ - f₁ -> X₂ - f₂ -> X₃
|          |          |
i₁         i₂         i₃
↓          ↓          ↓
Y₁ - g₁ -> Y₂ - g₂ -> Y₃
```
Then the left square is a pullback if the right square and the big square are.
-/
def leftSquareIsPullback (H : IsLimit t₂) (H' : IsLimit (t₂.pasteHoriz t₁ hi₂)) : IsLimit t₁ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X₃ Y₁ Y₂ Y₃ : C
    g₁ : Quiver.Hom Y₁ Y₂
    g₂ : Quiver.Hom Y₂ Y₃
    i₃ : Quiver.Hom X₃ Y₃
    t₂ : CategoryTheory.Limits.PullbackCone g₂ i₃
    i₂ : Quiver.Hom t₂.pt Y₂
    t₁ : CategoryTheory.Limits.PullbackCone g₁ i₂
    hi₂ : Eq i₂ t₂.fst
    H : CategoryTheory.Limits.IsLimit t₂
    H' : CategoryTheory.Limits.IsLimit (t₂.pasteHoriz t₁ hi₂)
    ⊢ CategoryTheory.Limits.IsLimit t₁
  -/
  apply PullbackCone.isLimitAux'
  /-
    case create
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X₃ Y₁ Y₂ Y₃ : C
    g₁ : Quiver.Hom Y₁ Y₂
    g₂ : Quiver.Hom Y₂ Y₃
    i₃ : Quiver.Hom X₃ Y₃
    t₂ : CategoryTheory.Limits.PullbackCone g₂ i₃
    i₂ : Quiver.Hom t₂.pt Y₂
    t₁ : CategoryTheory.Limits.PullbackCone g₁ i₂
    hi₂ : Eq i₂ t₂.fst
    H : CategoryTheory.Limits.IsLimit t₂
    H' : CategoryTheory.Limits.IsLimit (t₂.pasteHoriz t₁ hi₂)
    ⊢ (s : CategoryTheory.Limits.PullbackCone g₁ i₂) → Subtype fun l => And (Eq (C …
  -/
  intro s
  -- Obtain the induced morphism from the universal property of the big square
  obtain ⟨l, hl, hl'⟩ := PullbackCone.IsLimit.lift' H' s.fst (s.snd ≫ f₂)
    (by rw [Category.assoc, ← t₂.condition, reassoc_of% s.condition, ← hi₂])
  /-
    case create.mk.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X₃ Y₁ Y₂ Y₃ : C
    g₁ : Quiver.Hom Y₁ Y₂
    g₂ : Quiver.Hom Y₂ Y₃
    i₃ : Quiver.Hom X₃ Y₃
    t₂ : CategoryTheory.Limits.PullbackCone g₂ i₃
    i₂ : Quiver.Hom t₂.pt Y₂
    t₁ : CategoryTheory.Limits.PullbackCone g₁ i₂
    hi₂ : Eq i₂ t₂.fst
    H : CategoryTheory.Limits.IsLimit t₂
    H' : CategoryTheory.Limits.IsLimit (t₂.pasteHoriz t₁ hi₂)
    s : CategoryTheory.Limits.PullbackCone g₁ i₂
    l : Quiver.Hom s.pt (t₂.pasteHoriz t₁ hi₂).pt
    hl : Eq (CategoryTheory.CategoryStruct.comp l (t₂.pasteHoriz t₁ hi₂).fst) s.fst
    hl' : Eq (CategoryTheory.CategoryStruct.comp l (t₂.pasteHoriz t₁ hi₂).snd) (Ca …
    ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp l t₁.fst) s.fst …
  -/
  refine ⟨l, hl, ?_, ?_⟩
  -- To check that `l` is compatible with the projections, we use the universal property of `t₂`
    /-
      case create.mk.intro.refine_1
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X₃ Y₁ Y₂ Y₃ : C
      g₁ : Quiver.Hom Y₁ Y₂
      g₂ : Quiver.Hom Y₂ Y₃
      i₃ : Quiver.Hom X₃ Y₃
      t₂ : CategoryTheory.Limits.PullbackCone g₂ i₃
      i₂ : Quiver.Hom t₂.pt Y₂
      t₁ : CategoryTheory.Limits.PullbackCone g₁ i₂
      hi₂ : Eq i₂ t₂.fst
      H : CategoryTheory.Limits.IsLimit t₂
      H' : CategoryTheory.Limits.IsLimit (t₂.pasteHoriz t₁ hi₂)
      s : CategoryTheory.Limits.PullbackCone g₁ i₂
      l : Quiver.Hom s.pt (t₂.pasteHoriz t₁ hi₂).pt
      hl : Eq (CategoryTheory.CategoryStruct.comp l (t₂.pasteHoriz t₁ hi₂).fst) s.fst
      hl' : Eq (CategoryTheory.CategoryStruct.comp l (t₂.pasteHoriz t₁ hi₂).snd) (Ca …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp l t₁.snd) s.snd
    -/
  · apply PullbackCone.IsLimit.hom_ext H
      /-
        case create.mk.intro.refine_1.h₀
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X₃ Y₁ Y₂ Y₃ : C
        g₁ : Quiver.Hom Y₁ Y₂
        g₂ : Quiver.Hom Y₂ Y₃
        i₃ : Quiver.Hom X₃ Y₃
        t₂ : CategoryTheory.Limits.PullbackCone g₂ i₃
        i₂ : Quiver.Hom t₂.pt Y₂
        t₁ : CategoryTheory.Limits.PullbackCone g₁ i₂
        hi₂ : Eq i₂ t₂.fst
        H : CategoryTheory.Limits.IsLimit t₂
        H' : CategoryTheory.Limits.IsLimit (t₂.pasteHoriz t₁ hi₂)
        s : CategoryTheory.Limits.PullbackCone g₁ i₂
        l : Quiver.Hom s.pt (t₂.pasteHoriz t₁ hi₂).pt
        hl : Eq (CategoryTheory.CategoryStruct.comp l (t₂.pasteHoriz t₁ hi₂).fst) s.fst
        hl' : Eq (CategoryTheory.CategoryStruct.comp l (t₂.pasteHoriz t₁ hi₂).snd) (Ca …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp l …
      -/
    · simp [← s.condition, ← hl, ← t₁.condition, ← hi₂]
      /-
        🎉 no goals
      -/
      /-
        case create.mk.intro.refine_1.h₁
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X₃ Y₁ Y₂ Y₃ : C
        g₁ : Quiver.Hom Y₁ Y₂
        g₂ : Quiver.Hom Y₂ Y₃
        i₃ : Quiver.Hom X₃ Y₃
        t₂ : CategoryTheory.Limits.PullbackCone g₂ i₃
        i₂ : Quiver.Hom t₂.pt Y₂
        t₁ : CategoryTheory.Limits.PullbackCone g₁ i₂
        hi₂ : Eq i₂ t₂.fst
        H : CategoryTheory.Limits.IsLimit t₂
        H' : CategoryTheory.Limits.IsLimit (t₂.pasteHoriz t₁ hi₂)
        s : CategoryTheory.Limits.PullbackCone g₁ i₂
        l : Quiver.Hom s.pt (t₂.pasteHoriz t₁ hi₂).pt
        hl : Eq (CategoryTheory.CategoryStruct.comp l (t₂.pasteHoriz t₁ hi₂).fst) s.fst
        hl' : Eq (CategoryTheory.CategoryStruct.comp l (t₂.pasteHoriz t₁ hi₂).snd) (Ca …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp l …
      -/
    · simpa using hl'
      /-
        🎉 no goals
      -/
  -- Uniqueness of the lift follows from the universal property of the big square
    /-
      case create.mk.intro.refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X₃ Y₁ Y₂ Y₃ : C
      g₁ : Quiver.Hom Y₁ Y₂
      g₂ : Quiver.Hom Y₂ Y₃
      i₃ : Quiver.Hom X₃ Y₃
      t₂ : CategoryTheory.Limits.PullbackCone g₂ i₃
      i₂ : Quiver.Hom t₂.pt Y₂
      t₁ : CategoryTheory.Limits.PullbackCone g₁ i₂
      hi₂ : Eq i₂ t₂.fst
      H : CategoryTheory.Limits.IsLimit t₂
      H' : CategoryTheory.Limits.IsLimit (t₂.pasteHoriz t₁ hi₂)
      s : CategoryTheory.Limits.PullbackCone g₁ i₂
      l : Quiver.Hom s.pt (t₂.pasteHoriz t₁ hi₂).pt
      hl : Eq (CategoryTheory.CategoryStruct.comp l (t₂.pasteHoriz t₁ hi₂).fst) s.fst
      hl' : Eq (CategoryTheory.CategoryStruct.comp l (t₂.pasteHoriz t₁ hi₂).snd) (Ca …
      ⊢ ∀ {m : Quiver.Hom s.pt t₁.pt}, Eq (CategoryTheory.CategoryStruct.comp m t₁.f …
    -/
  · intro m hm₁ hm₂
    /-
      case create.mk.intro.refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X₃ Y₁ Y₂ Y₃ : C
      g₁ : Quiver.Hom Y₁ Y₂
      g₂ : Quiver.Hom Y₂ Y₃
      i₃ : Quiver.Hom X₃ Y₃
      t₂ : CategoryTheory.Limits.PullbackCone g₂ i₃
      i₂ : Quiver.Hom t₂.pt Y₂
      t₁ : CategoryTheory.Limits.PullbackCone g₁ i₂
      hi₂ : Eq i₂ t₂.fst
      H : CategoryTheory.Limits.IsLimit t₂
      H' : CategoryTheory.Limits.IsLimit (t₂.pasteHoriz t₁ hi₂)
      s : CategoryTheory.Limits.PullbackCone g₁ i₂
      l : Quiver.Hom s.pt (t₂.pasteHoriz t₁ hi₂).pt
      hl : Eq (CategoryTheory.CategoryStruct.comp l (t₂.pasteHoriz t₁ hi₂).fst) s.fst
      hl' : Eq (CategoryTheory.CategoryStruct.comp l (t₂.pasteHoriz t₁ hi₂).snd) (Ca …
      m : Quiver.Hom s.pt t₁.pt
      hm₁ : Eq (CategoryTheory.CategoryStruct.comp m t₁.fst) s.fst
      hm₂ : Eq (CategoryTheory.CategoryStruct.comp m t₁.snd) s.snd
      ⊢ Eq m l
    -/
    apply PullbackCone.IsLimit.hom_ext H'
      /-
        case create.mk.intro.refine_2.h₀
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X₃ Y₁ Y₂ Y₃ : C
        g₁ : Quiver.Hom Y₁ Y₂
        g₂ : Quiver.Hom Y₂ Y₃
        i₃ : Quiver.Hom X₃ Y₃
        t₂ : CategoryTheory.Limits.PullbackCone g₂ i₃
        i₂ : Quiver.Hom t₂.pt Y₂
        t₁ : CategoryTheory.Limits.PullbackCone g₁ i₂
        hi₂ : Eq i₂ t₂.fst
        H : CategoryTheory.Limits.IsLimit t₂
        H' : CategoryTheory.Limits.IsLimit (t₂.pasteHoriz t₁ hi₂)
        s : CategoryTheory.Limits.PullbackCone g₁ i₂
        l : Quiver.Hom s.pt (t₂.pasteHoriz t₁ hi₂).pt
        hl : Eq (CategoryTheory.CategoryStruct.comp l (t₂.pasteHoriz t₁ hi₂).fst) s.fst
        hl' : Eq (CategoryTheory.CategoryStruct.comp l (t₂.pasteHoriz t₁ hi₂).snd) (Ca …
        m : Quiver.Hom s.pt t₁.pt
        hm₁ : Eq (CategoryTheory.CategoryStruct.comp m t₁.fst) s.fst
        hm₂ : Eq (CategoryTheory.CategoryStruct.comp m t₁.snd) s.snd
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m (t₂.pasteHoriz t₁ hi₂).fst) (Catego …
      -/
    · simpa [hm₁] using hl.symm
      /-
        🎉 no goals
      -/
      /-
        case create.mk.intro.refine_2.h₁
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X₃ Y₁ Y₂ Y₃ : C
        g₁ : Quiver.Hom Y₁ Y₂
        g₂ : Quiver.Hom Y₂ Y₃
        i₃ : Quiver.Hom X₃ Y₃
        t₂ : CategoryTheory.Limits.PullbackCone g₂ i₃
        i₂ : Quiver.Hom t₂.pt Y₂
        t₁ : CategoryTheory.Limits.PullbackCone g₁ i₂
        hi₂ : Eq i₂ t₂.fst
        H : CategoryTheory.Limits.IsLimit t₂
        H' : CategoryTheory.Limits.IsLimit (t₂.pasteHoriz t₁ hi₂)
        s : CategoryTheory.Limits.PullbackCone g₁ i₂
        l : Quiver.Hom s.pt (t₂.pasteHoriz t₁ hi₂).pt
        hl : Eq (CategoryTheory.CategoryStruct.comp l (t₂.pasteHoriz t₁ hi₂).fst) s.fst
        hl' : Eq (CategoryTheory.CategoryStruct.comp l (t₂.pasteHoriz t₁ hi₂).snd) (Ca …
        m : Quiver.Hom s.pt t₁.pt
        hm₁ : Eq (CategoryTheory.CategoryStruct.comp m t₁.fst) s.fst
        hm₂ : Eq (CategoryTheory.CategoryStruct.comp m t₁.snd) s.snd
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m (t₂.pasteHoriz t₁ hi₂).snd) (Catego …
      -/
    · simpa [← hm₂] using hl'.symm
      /-
        🎉 no goals
      -/


/-- Given that the right square is a pullback, the pasted square is a pullback iff the left
square is. -/
def pasteHorizIsPullbackEquiv (H : IsLimit t₂) : IsLimit (t₂.pasteHoriz t₁ hi₂) ≃ IsLimit t₁ where
  toFun H' := leftSquareIsPullback t₁ _ H H'
  invFun H' := pasteHorizIsPullback _ H H'
  left_inv _ := Subsingleton.elim _ _
  right_inv _ := Subsingleton.elim _ _


/-- The `PullbackCone` obtained by pasting two `PullbackCone`'s vertically -/
abbrev PullbackCone.pasteVert
    (t₁ : PullbackCone i₁ f₁) {i₂ : t₁.pt ⟶ X₂} (t₂ : PullbackCone i₂ f₂) (hi₂ : i₂ = t₁.snd) :
    PullbackCone i₁ (f₂ ≫ f₁) :=
  PullbackCone.mk (t₂.fst ≫ t₁.fst) t₂.snd
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X₁ X₂ X₃ Y₁ : C
          f₁ : Quiver.Hom X₂ X₁
          f₂ : Quiver.Hom X₃ X₂
          i₁ : Quiver.Hom Y₁ X₁
          t₁ : CategoryTheory.Limits.PullbackCone i₁ f₁
          i₂ : Quiver.Hom t₁.pt X₂
          t₂ : CategoryTheory.Limits.PullbackCone i₂ f₂
          hi₂ : Eq i₂ t₁.snd
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp t …
        -/
    (by rw [← reassoc_of% t₂.condition, Category.assoc, t₁.condition, ← hi₂])
        /-
          🎉 no goals
        -/


local notation "Y₂" => t₁.pt

local notation "g₁" => t₁.fst

local notation "i₂" => t₁.snd

local notation "Y₃" => t₂.pt

local notation "g₂" => t₂.fst

local notation "i₃" => t₂.snd


/-- Pasting two pullback cones vertically is isomorphic to the pullback cone obtained by flipping
them, pasting horizontally, and then flipping the result again. -/
def PullbackCone.pasteVertFlip : (t₁.pasteVert t₂ hi₂).flip ≅ (t₁.flip.pasteHoriz t₂.flip hi₂) :=
                                    /-
                                      C : Type u
                                      inst✝ : CategoryTheory.Category.{v, u} C
                                      X₁ X₂ X₃ Y₁ : C
                                      f₁ : Quiver.Hom X₂ X₁
                                      f₂ : Quiver.Hom X₃ X₂
                                      i₁ : Quiver.Hom Y₁ X₁
                                      t₁ : CategoryTheory.Limits.PullbackCone i₁ f₁
                                      i₂ : Quiver.Hom t₁.pt X₂
                                      t₂ : CategoryTheory.Limits.PullbackCone «i₂» f₂
                                      hi₂ : Eq «i₂» t₁.snd
                                      ⊢ Eq (t₁.pasteVert t₂ hi₂).flip.fst (CategoryTheory.CategoryStruct.comp (Categ …
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
  PullbackCone.ext (Iso.refl _) (by simp) (by simp)
                                              /-
                                                🎉 no goals
                                              -/


/-- Given
```
Y₃ - i₃ -> X₃
|          |
g₂         f₂
∨          ∨
Y₂ - i₂ -> X₂
|          |
g₁         f₁
∨          ∨
Y₁ - i₁ -> X₁
```
The big square is a pullback if both the small squares are.
-/
def pasteVertIsPullback (H₁ : IsLimit t₁) (H₂ : IsLimit t₂) : IsLimit (t₁.pasteVert t₂ hi₂) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Y₁ : C
    f₁ : Quiver.Hom X₂ X₁
    f₂ : Quiver.Hom X₃ X₂
    i₁ : Quiver.Hom Y₁ X₁
    t₁ : CategoryTheory.Limits.PullbackCone i₁ f₁
    i₂ : Quiver.Hom t₁.pt X₂
    t₂ : CategoryTheory.Limits.PullbackCone «i₂» f₂
    hi₂ : Eq «i₂» t₁.snd
    H₁ : CategoryTheory.Limits.IsLimit t₁
    H₂ : CategoryTheory.Limits.IsLimit t₂
    ⊢ CategoryTheory.Limits.IsLimit (t₁.pasteVert t₂ hi₂)
  -/
  apply PullbackCone.isLimitOfFlip <| IsLimit.ofIsoLimit _ (t₁.pasteVertFlip t₂ hi₂).symm
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Y₁ : C
    f₁ : Quiver.Hom X₂ X₁
    f₂ : Quiver.Hom X₃ X₂
    i₁ : Quiver.Hom Y₁ X₁
    t₁ : CategoryTheory.Limits.PullbackCone i₁ f₁
    i₂ : Quiver.Hom t₁.pt X₂
    t₂ : CategoryTheory.Limits.PullbackCone «i₂» f₂
    hi₂ : Eq «i₂» t₁.snd
    H₁ : CategoryTheory.Limits.IsLimit t₁
    H₂ : CategoryTheory.Limits.IsLimit t₂
    ⊢ CategoryTheory.Limits.IsLimit (t₁.flip.pasteHoriz t₂.flip hi₂)
  -/
  exact pasteHorizIsPullback hi₂ (PullbackCone.flipIsLimit H₁) (PullbackCone.flipIsLimit H₂)
  /-
    🎉 no goals
  -/


/-- Given
```
Y₃ - i₃ -> X₃
|          |
g₂         f₂
∨          ∨
Y₂ - i₂ -> X₂
|          |
g₁         f₁
∨          ∨
Y₁ - i₁ -> X₁
```
The top square is a pullback if the bottom square and the big square are.
-/
def topSquareIsPullback (H₁ : IsLimit t₁) (H₂ : IsLimit (t₁.pasteVert t₂ hi₂)) : IsLimit t₂ :=
  PullbackCone.isLimitOfFlip
    (leftSquareIsPullback _ hi₂ (PullbackCone.flipIsLimit H₁) (PullbackCone.flipIsLimit H₂))


/-- Given that the bottom square is a pullback, the pasted square is a pullback iff the top
square is. -/
def pasteVertIsPullbackEquiv (H : IsLimit t₁) : IsLimit (t₁.pasteVert t₂ hi₂) ≃ IsLimit t₂ where
  toFun H' := topSquareIsPullback t₂ _ H H'
  invFun H' := pasteVertIsPullback _ H H'
  left_inv _ := Subsingleton.elim _ _
  right_inv _ := Subsingleton.elim _ _


/-- The pushout cocone obtained by pasting two pushout cocones horizontally. -/
abbrev PushoutCocone.pasteHoriz
    (t₁ : PushoutCocone i₁ f₁) {i₂ : X₂ ⟶ t₁.pt} (t₂ : PushoutCocone i₂ f₂) (hi₂ : i₂ = t₁.inr) :
    PushoutCocone i₁ (f₁ ≫ f₂) :=
  PushoutCocone.mk (t₁.inl ≫ t₂.inl) t₂.inr
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X₁ X₂ X₃ Y₁ : C
          f₁ : Quiver.Hom X₁ X₂
          f₂ : Quiver.Hom X₂ X₃
          i₁ : Quiver.Hom X₁ Y₁
          t₁ : CategoryTheory.Limits.PushoutCocone i₁ f₁
          i₂ : Quiver.Hom X₂ t₁.pt
          t₂ : CategoryTheory.Limits.PushoutCocone i₂ f₂
          hi₂ : Eq i₂ t₁.inr
          ⊢ Eq (CategoryTheory.CategoryStruct.comp i₁ (CategoryTheory.CategoryStruct.com …
        -/
    (by rw [reassoc_of% t₁.condition, Category.assoc, ← t₂.condition, ← hi₂])
        /-
          🎉 no goals
        -/


local notation "g₁" => t₁.inl

local notation "i₂" => t₁.inr

local notation "g₂" => t₂.inl

local notation "i₃" => t₂.inr


/-- Given
```
X₁ - f₁ -> X₂ - f₂ -> X₃
|          |          |
i₁         i₂         i₃
∨          ∨          ∨
Y₁ - g₁ -> Y₂ - g₂ -> Y₃
```
Then the big square is a pushout if both the small squares are.
-/
def pasteHorizIsPushout (H : IsColimit t₁) (H' : IsColimit t₂) :
    IsColimit (t₁.pasteHoriz t₂ hi₂) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Y₁ : C
    f₁ : Quiver.Hom X₁ X₂
    f₂ : Quiver.Hom X₂ X₃
    i₁ : Quiver.Hom X₁ Y₁
    t₁ : CategoryTheory.Limits.PushoutCocone i₁ f₁
    i₂ : Quiver.Hom X₂ t₁.pt
    t₂ : CategoryTheory.Limits.PushoutCocone «i₂» f₂
    hi₂ : Eq «i₂» t₁.inr
    H : CategoryTheory.Limits.IsColimit t₁
    H' : CategoryTheory.Limits.IsColimit t₂
    ⊢ CategoryTheory.Limits.IsColimit (t₁.pasteHoriz t₂ hi₂)
  -/
  apply PushoutCocone.isColimitAux'
  /-
    case create
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Y₁ : C
    f₁ : Quiver.Hom X₁ X₂
    f₂ : Quiver.Hom X₂ X₃
    i₁ : Quiver.Hom X₁ Y₁
    t₁ : CategoryTheory.Limits.PushoutCocone i₁ f₁
    i₂ : Quiver.Hom X₂ t₁.pt
    t₂ : CategoryTheory.Limits.PushoutCocone «i₂» f₂
    hi₂ : Eq «i₂» t₁.inr
    H : CategoryTheory.Limits.IsColimit t₁
    H' : CategoryTheory.Limits.IsColimit t₂
    ⊢ (s : CategoryTheory.Limits.PushoutCocone i₁ (CategoryTheory.CategoryStruct.c …
  -/
  intro s
  -- Obtain the induced map from descending from both the small squares consecutively.
  obtain ⟨l₁, hl₁, hl₁'⟩ := PushoutCocone.IsColimit.desc' H s.inl (f₂ ≫ s.inr)
    (by rw [s.condition, Category.assoc])
  /-
    case create.mk.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Y₁ : C
    f₁ : Quiver.Hom X₁ X₂
    f₂ : Quiver.Hom X₂ X₃
    i₁ : Quiver.Hom X₁ Y₁
    t₁ : CategoryTheory.Limits.PushoutCocone i₁ f₁
    i₂ : Quiver.Hom X₂ t₁.pt
    t₂ : CategoryTheory.Limits.PushoutCocone «i₂» f₂
    hi₂ : Eq «i₂» t₁.inr
    H : CategoryTheory.Limits.IsColimit t₁
    H' : CategoryTheory.Limits.IsColimit t₂
    s : CategoryTheory.Limits.PushoutCocone i₁ (CategoryTheory.CategoryStruct.comp …
    l₁ : Quiver.Hom t₁.pt s.pt
    hl₁ : Eq (CategoryTheory.CategoryStruct.comp t₁.inl l₁) s.inl
    hl₁' : Eq (CategoryTheory.CategoryStruct.comp t₁.inr l₁) (CategoryTheory.Categ …
    ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp (t₁.pasteHoriz  …
  -/
  obtain ⟨l₂, hl₂, hl₂'⟩ := PushoutCocone.IsColimit.desc' H' l₁ s.inr (by rw [← hl₁', hi₂])
  /-
    case create.mk.intro.mk.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Y₁ : C
    f₁ : Quiver.Hom X₁ X₂
    f₂ : Quiver.Hom X₂ X₃
    i₁ : Quiver.Hom X₁ Y₁
    t₁ : CategoryTheory.Limits.PushoutCocone i₁ f₁
    i₂ : Quiver.Hom X₂ t₁.pt
    t₂ : CategoryTheory.Limits.PushoutCocone «i₂» f₂
    hi₂ : Eq «i₂» t₁.inr
    H : CategoryTheory.Limits.IsColimit t₁
    H' : CategoryTheory.Limits.IsColimit t₂
    s : CategoryTheory.Limits.PushoutCocone i₁ (CategoryTheory.CategoryStruct.comp …
    l₁ : Quiver.Hom t₁.pt s.pt
    hl₁ : Eq (CategoryTheory.CategoryStruct.comp t₁.inl l₁) s.inl
    hl₁' : Eq (CategoryTheory.CategoryStruct.comp t₁.inr l₁) (CategoryTheory.Categ …
    l₂ : Quiver.Hom t₂.pt s.pt
    hl₂ : Eq (CategoryTheory.CategoryStruct.comp t₂.inl l₂) l₁
    hl₂' : Eq (CategoryTheory.CategoryStruct.comp t₂.inr l₂) s.inr
    ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp (t₁.pasteHoriz  …
  -/
  refine ⟨l₂, by simp [hl₂, hl₁], hl₂', ?_⟩
  -- Uniqueness also follows from the universal property of both the small squares.
  /-
    case create.mk.intro.mk.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Y₁ : C
    f₁ : Quiver.Hom X₁ X₂
    f₂ : Quiver.Hom X₂ X₃
    i₁ : Quiver.Hom X₁ Y₁
    t₁ : CategoryTheory.Limits.PushoutCocone i₁ f₁
    i₂ : Quiver.Hom X₂ t₁.pt
    t₂ : CategoryTheory.Limits.PushoutCocone «i₂» f₂
    hi₂ : Eq «i₂» t₁.inr
    H : CategoryTheory.Limits.IsColimit t₁
    H' : CategoryTheory.Limits.IsColimit t₂
    s : CategoryTheory.Limits.PushoutCocone i₁ (CategoryTheory.CategoryStruct.comp …
    l₁ : Quiver.Hom t₁.pt s.pt
    hl₁ : Eq (CategoryTheory.CategoryStruct.comp t₁.inl l₁) s.inl
    hl₁' : Eq (CategoryTheory.CategoryStruct.comp t₁.inr l₁) (CategoryTheory.Categ …
    l₂ : Quiver.Hom t₂.pt s.pt
    hl₂ : Eq (CategoryTheory.CategoryStruct.comp t₂.inl l₂) l₁
    hl₂' : Eq (CategoryTheory.CategoryStruct.comp t₂.inr l₂) s.inr
    ⊢ ∀ {m : Quiver.Hom (t₁.pasteHoriz t₂ hi₂).pt s.pt}, Eq (CategoryTheory.Catego …
  -/
  intro m hm₁ hm₂
  /-
    case create.mk.intro.mk.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Y₁ : C
    f₁ : Quiver.Hom X₁ X₂
    f₂ : Quiver.Hom X₂ X₃
    i₁ : Quiver.Hom X₁ Y₁
    t₁ : CategoryTheory.Limits.PushoutCocone i₁ f₁
    i₂ : Quiver.Hom X₂ t₁.pt
    t₂ : CategoryTheory.Limits.PushoutCocone «i₂» f₂
    hi₂ : Eq «i₂» t₁.inr
    H : CategoryTheory.Limits.IsColimit t₁
    H' : CategoryTheory.Limits.IsColimit t₂
    s : CategoryTheory.Limits.PushoutCocone i₁ (CategoryTheory.CategoryStruct.comp …
    l₁ : Quiver.Hom t₁.pt s.pt
    hl₁ : Eq (CategoryTheory.CategoryStruct.comp t₁.inl l₁) s.inl
    hl₁' : Eq (CategoryTheory.CategoryStruct.comp t₁.inr l₁) (CategoryTheory.Categ …
    l₂ : Quiver.Hom t₂.pt s.pt
    hl₂ : Eq (CategoryTheory.CategoryStruct.comp t₂.inl l₂) l₁
    hl₂' : Eq (CategoryTheory.CategoryStruct.comp t₂.inr l₂) s.inr
    m : Quiver.Hom (t₁.pasteHoriz t₂ hi₂).pt s.pt
    hm₁ : Eq (CategoryTheory.CategoryStruct.comp (t₁.pasteHoriz t₂ hi₂).inl m) s.inl
    hm₂ : Eq (CategoryTheory.CategoryStruct.comp (t₁.pasteHoriz t₂ hi₂).inr m) s.inr
    ⊢ Eq m l₂
  -/
  apply PushoutCocone.IsColimit.hom_ext H' _ (by simpa [hl₂'] using hm₂)
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Y₁ : C
    f₁ : Quiver.Hom X₁ X₂
    f₂ : Quiver.Hom X₂ X₃
    i₁ : Quiver.Hom X₁ Y₁
    t₁ : CategoryTheory.Limits.PushoutCocone i₁ f₁
    i₂ : Quiver.Hom X₂ t₁.pt
    t₂ : CategoryTheory.Limits.PushoutCocone «i₂» f₂
    hi₂ : Eq «i₂» t₁.inr
    H : CategoryTheory.Limits.IsColimit t₁
    H' : CategoryTheory.Limits.IsColimit t₂
    s : CategoryTheory.Limits.PushoutCocone i₁ (CategoryTheory.CategoryStruct.comp …
    l₁ : Quiver.Hom t₁.pt s.pt
    hl₁ : Eq (CategoryTheory.CategoryStruct.comp t₁.inl l₁) s.inl
    hl₁' : Eq (CategoryTheory.CategoryStruct.comp t₁.inr l₁) (CategoryTheory.Categ …
    l₂ : Quiver.Hom t₂.pt s.pt
    hl₂ : Eq (CategoryTheory.CategoryStruct.comp t₂.inl l₂) l₁
    hl₂' : Eq (CategoryTheory.CategoryStruct.comp t₂.inr l₂) s.inr
    m : Quiver.Hom (t₁.pasteHoriz t₂ hi₂).pt s.pt
    hm₁ : Eq (CategoryTheory.CategoryStruct.comp (t₁.pasteHoriz t₂ hi₂).inl m) s.inl
    hm₂ : Eq (CategoryTheory.CategoryStruct.comp (t₁.pasteHoriz t₂ hi₂).inr m) s.inr
    ⊢ Eq (CategoryTheory.CategoryStruct.comp t₂.inl m) (CategoryTheory.CategoryStr …
  -/
  simp only [PushoutCocone.mk_pt, PushoutCocone.mk_ι_app, Category.assoc] at hm₁ hm₂
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Y₁ : C
    f₁ : Quiver.Hom X₁ X₂
    f₂ : Quiver.Hom X₂ X₃
    i₁ : Quiver.Hom X₁ Y₁
    t₁ : CategoryTheory.Limits.PushoutCocone i₁ f₁
    i₂ : Quiver.Hom X₂ t₁.pt
    t₂ : CategoryTheory.Limits.PushoutCocone «i₂» f₂
    hi₂ : Eq «i₂» t₁.inr
    H : CategoryTheory.Limits.IsColimit t₁
    H' : CategoryTheory.Limits.IsColimit t₂
    s : CategoryTheory.Limits.PushoutCocone i₁ (CategoryTheory.CategoryStruct.comp …
    l₁ : Quiver.Hom t₁.pt s.pt
    hl₁ : Eq (CategoryTheory.CategoryStruct.comp t₁.inl l₁) s.inl
    hl₁' : Eq (CategoryTheory.CategoryStruct.comp t₁.inr l₁) (CategoryTheory.Categ …
    l₂ : Quiver.Hom t₂.pt s.pt
    hl₂ : Eq (CategoryTheory.CategoryStruct.comp t₂.inl l₂) l₁
    hl₂' : Eq (CategoryTheory.CategoryStruct.comp t₂.inr l₂) s.inr
    m : Quiver.Hom (t₁.pasteHoriz t₂ hi₂).pt s.pt
    hm₂ : Eq (CategoryTheory.CategoryStruct.comp t₂.inr m) s.inr
    hm₁ : Eq (CategoryTheory.CategoryStruct.comp t₁.inl (CategoryTheory.CategorySt …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp t₂.inl m) (CategoryTheory.CategoryStr …
  -/
  apply PushoutCocone.IsColimit.hom_ext H
    /-
      case h₀
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X₁ X₂ X₃ Y₁ : C
      f₁ : Quiver.Hom X₁ X₂
      f₂ : Quiver.Hom X₂ X₃
      i₁ : Quiver.Hom X₁ Y₁
      t₁ : CategoryTheory.Limits.PushoutCocone i₁ f₁
      i₂ : Quiver.Hom X₂ t₁.pt
      t₂ : CategoryTheory.Limits.PushoutCocone «i₂» f₂
      hi₂ : Eq «i₂» t₁.inr
      H : CategoryTheory.Limits.IsColimit t₁
      H' : CategoryTheory.Limits.IsColimit t₂
      s : CategoryTheory.Limits.PushoutCocone i₁ (CategoryTheory.CategoryStruct.comp …
      l₁ : Quiver.Hom t₁.pt s.pt
      hl₁ : Eq (CategoryTheory.CategoryStruct.comp t₁.inl l₁) s.inl
      hl₁' : Eq (CategoryTheory.CategoryStruct.comp t₁.inr l₁) (CategoryTheory.Categ …
      l₂ : Quiver.Hom t₂.pt s.pt
      hl₂ : Eq (CategoryTheory.CategoryStruct.comp t₂.inl l₂) l₁
      hl₂' : Eq (CategoryTheory.CategoryStruct.comp t₂.inr l₂) s.inr
      m : Quiver.Hom (t₁.pasteHoriz t₂ hi₂).pt s.pt
      hm₂ : Eq (CategoryTheory.CategoryStruct.comp t₂.inr m) s.inr
      hm₁ : Eq (CategoryTheory.CategoryStruct.comp t₁.inl (CategoryTheory.CategorySt …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp t₁.inl (CategoryTheory.CategoryStruct …
    -/
  · rw [hm₁, ← hl₁, hl₂]
    /-
      🎉 no goals
    -/
    /-
      case h₁
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X₁ X₂ X₃ Y₁ : C
      f₁ : Quiver.Hom X₁ X₂
      f₂ : Quiver.Hom X₂ X₃
      i₁ : Quiver.Hom X₁ Y₁
      t₁ : CategoryTheory.Limits.PushoutCocone i₁ f₁
      i₂ : Quiver.Hom X₂ t₁.pt
      t₂ : CategoryTheory.Limits.PushoutCocone «i₂» f₂
      hi₂ : Eq «i₂» t₁.inr
      H : CategoryTheory.Limits.IsColimit t₁
      H' : CategoryTheory.Limits.IsColimit t₂
      s : CategoryTheory.Limits.PushoutCocone i₁ (CategoryTheory.CategoryStruct.comp …
      l₁ : Quiver.Hom t₁.pt s.pt
      hl₁ : Eq (CategoryTheory.CategoryStruct.comp t₁.inl l₁) s.inl
      hl₁' : Eq (CategoryTheory.CategoryStruct.comp t₁.inr l₁) (CategoryTheory.Categ …
      l₂ : Quiver.Hom t₂.pt s.pt
      hl₂ : Eq (CategoryTheory.CategoryStruct.comp t₂.inl l₂) l₁
      hl₂' : Eq (CategoryTheory.CategoryStruct.comp t₂.inr l₂) s.inr
      m : Quiver.Hom (t₁.pasteHoriz t₂ hi₂).pt s.pt
      hm₂ : Eq (CategoryTheory.CategoryStruct.comp t₂.inr m) s.inr
      hm₁ : Eq (CategoryTheory.CategoryStruct.comp t₁.inl (CategoryTheory.CategorySt …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp t₁.inr (CategoryTheory.CategoryStruct …
    -/
  · rw [← hi₂, reassoc_of% t₂.condition, reassoc_of% t₂.condition, hm₂, hl₂']
    /-
      🎉 no goals
    -/


/-- Given

X₁ - f₁ -> X₂ - f₂ -> X₃
|          |          |
i₁         i₂         i₃
∨          ∨          ∨
Y₁ - g₁ -> Y₂ - g₂ -> Y₃

Then the right square is a pushout if the left square and the big square are.
-/
def rightSquareIsPushout (H : IsColimit t₁) (H' : IsColimit (t₁.pasteHoriz t₂ hi₂)) :
    IsColimit t₂ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Y₁ : C
    f₁ : Quiver.Hom X₁ X₂
    f₂ : Quiver.Hom X₂ X₃
    i₁ : Quiver.Hom X₁ Y₁
    t₁ : CategoryTheory.Limits.PushoutCocone i₁ f₁
    i₂ : Quiver.Hom X₂ t₁.pt
    t₂ : CategoryTheory.Limits.PushoutCocone «i₂» f₂
    hi₂ : Eq «i₂» t₁.inr
    H : CategoryTheory.Limits.IsColimit t₁
    H' : CategoryTheory.Limits.IsColimit (t₁.pasteHoriz t₂ hi₂)
    ⊢ CategoryTheory.Limits.IsColimit t₂
  -/
  apply PushoutCocone.isColimitAux'
  /-
    case create
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Y₁ : C
    f₁ : Quiver.Hom X₁ X₂
    f₂ : Quiver.Hom X₂ X₃
    i₁ : Quiver.Hom X₁ Y₁
    t₁ : CategoryTheory.Limits.PushoutCocone i₁ f₁
    i₂ : Quiver.Hom X₂ t₁.pt
    t₂ : CategoryTheory.Limits.PushoutCocone «i₂» f₂
    hi₂ : Eq «i₂» t₁.inr
    H : CategoryTheory.Limits.IsColimit t₁
    H' : CategoryTheory.Limits.IsColimit (t₁.pasteHoriz t₂ hi₂)
    ⊢ (s : CategoryTheory.Limits.PushoutCocone «i₂» f₂) → Subtype fun l => And (Eq …
  -/
  intro s
  -- Obtain the induced morphism from the universal property of the big square
  obtain ⟨l, hl, hl'⟩ := PushoutCocone.IsColimit.desc' H' (g₁ ≫ s.inl) s.inr
    (by rw [reassoc_of% t₁.condition, ← hi₂, s.condition, Category.assoc])
  /-
    case create.mk.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Y₁ : C
    f₁ : Quiver.Hom X₁ X₂
    f₂ : Quiver.Hom X₂ X₃
    i₁ : Quiver.Hom X₁ Y₁
    t₁ : CategoryTheory.Limits.PushoutCocone i₁ f₁
    i₂ : Quiver.Hom X₂ t₁.pt
    t₂ : CategoryTheory.Limits.PushoutCocone «i₂» f₂
    hi₂ : Eq «i₂» t₁.inr
    H : CategoryTheory.Limits.IsColimit t₁
    H' : CategoryTheory.Limits.IsColimit (t₁.pasteHoriz t₂ hi₂)
    s : CategoryTheory.Limits.PushoutCocone «i₂» f₂
    l : Quiver.Hom (t₁.pasteHoriz t₂ hi₂).pt s.pt
    hl : Eq (CategoryTheory.CategoryStruct.comp (t₁.pasteHoriz t₂ hi₂).inl l) (Cat …
    hl' : Eq (CategoryTheory.CategoryStruct.comp (t₁.pasteHoriz t₂ hi₂).inr l) s.inr
    ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp t₂.inl l) s.inl …
  -/
  refine ⟨l, ?_, hl', ?_⟩
  -- To check that `l` is compatible with the projections, we use the universal property of `t₁`
    /-
      case create.mk.intro.refine_1
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X₁ X₂ X₃ Y₁ : C
      f₁ : Quiver.Hom X₁ X₂
      f₂ : Quiver.Hom X₂ X₃
      i₁ : Quiver.Hom X₁ Y₁
      t₁ : CategoryTheory.Limits.PushoutCocone i₁ f₁
      i₂ : Quiver.Hom X₂ t₁.pt
      t₂ : CategoryTheory.Limits.PushoutCocone «i₂» f₂
      hi₂ : Eq «i₂» t₁.inr
      H : CategoryTheory.Limits.IsColimit t₁
      H' : CategoryTheory.Limits.IsColimit (t₁.pasteHoriz t₂ hi₂)
      s : CategoryTheory.Limits.PushoutCocone «i₂» f₂
      l : Quiver.Hom (t₁.pasteHoriz t₂ hi₂).pt s.pt
      hl : Eq (CategoryTheory.CategoryStruct.comp (t₁.pasteHoriz t₂ hi₂).inl l) (Cat …
      hl' : Eq (CategoryTheory.CategoryStruct.comp (t₁.pasteHoriz t₂ hi₂).inr l) s.inr
      ⊢ Eq (CategoryTheory.CategoryStruct.comp t₂.inl l) s.inl
    -/
  · simp at hl hl'
    /-
      case create.mk.intro.refine_1
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X₁ X₂ X₃ Y₁ : C
      f₁ : Quiver.Hom X₁ X₂
      f₂ : Quiver.Hom X₂ X₃
      i₁ : Quiver.Hom X₁ Y₁
      t₁ : CategoryTheory.Limits.PushoutCocone i₁ f₁
      i₂ : Quiver.Hom X₂ t₁.pt
      t₂ : CategoryTheory.Limits.PushoutCocone «i₂» f₂
      hi₂ : Eq «i₂» t₁.inr
      H : CategoryTheory.Limits.IsColimit t₁
      H' : CategoryTheory.Limits.IsColimit (t₁.pasteHoriz t₂ hi₂)
      s : CategoryTheory.Limits.PushoutCocone «i₂» f₂
      l : Quiver.Hom (t₁.pasteHoriz t₂ hi₂).pt s.pt
      hl' : Eq (CategoryTheory.CategoryStruct.comp t₂.inr l) s.inr
      hl : Eq (CategoryTheory.CategoryStruct.comp t₁.inl (CategoryTheory.CategoryStr …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp t₂.inl l) s.inl
    -/
    apply PushoutCocone.IsColimit.hom_ext H hl
    /-
      case create.mk.intro.refine_1
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X₁ X₂ X₃ Y₁ : C
      f₁ : Quiver.Hom X₁ X₂
      f₂ : Quiver.Hom X₂ X₃
      i₁ : Quiver.Hom X₁ Y₁
      t₁ : CategoryTheory.Limits.PushoutCocone i₁ f₁
      i₂ : Quiver.Hom X₂ t₁.pt
      t₂ : CategoryTheory.Limits.PushoutCocone «i₂» f₂
      hi₂ : Eq «i₂» t₁.inr
      H : CategoryTheory.Limits.IsColimit t₁
      H' : CategoryTheory.Limits.IsColimit (t₁.pasteHoriz t₂ hi₂)
      s : CategoryTheory.Limits.PushoutCocone «i₂» f₂
      l : Quiver.Hom (t₁.pasteHoriz t₂ hi₂).pt s.pt
      hl' : Eq (CategoryTheory.CategoryStruct.comp t₂.inr l) s.inr
      hl : Eq (CategoryTheory.CategoryStruct.comp t₁.inl (CategoryTheory.CategoryStr …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp t₁.inr (CategoryTheory.CategoryStruct …
    -/
    rw [← Category.assoc, ← hi₂, t₂.condition, s.condition, Category.assoc, hl']
    /-
      🎉 no goals
    -/
  -- Uniqueness of the lift follows from the universal property of the big square
    /-
      case create.mk.intro.refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X₁ X₂ X₃ Y₁ : C
      f₁ : Quiver.Hom X₁ X₂
      f₂ : Quiver.Hom X₂ X₃
      i₁ : Quiver.Hom X₁ Y₁
      t₁ : CategoryTheory.Limits.PushoutCocone i₁ f₁
      i₂ : Quiver.Hom X₂ t₁.pt
      t₂ : CategoryTheory.Limits.PushoutCocone «i₂» f₂
      hi₂ : Eq «i₂» t₁.inr
      H : CategoryTheory.Limits.IsColimit t₁
      H' : CategoryTheory.Limits.IsColimit (t₁.pasteHoriz t₂ hi₂)
      s : CategoryTheory.Limits.PushoutCocone «i₂» f₂
      l : Quiver.Hom (t₁.pasteHoriz t₂ hi₂).pt s.pt
      hl : Eq (CategoryTheory.CategoryStruct.comp (t₁.pasteHoriz t₂ hi₂).inl l) (Cat …
      hl' : Eq (CategoryTheory.CategoryStruct.comp (t₁.pasteHoriz t₂ hi₂).inr l) s.inr
      ⊢ ∀ {m : Quiver.Hom t₂.pt s.pt}, Eq (CategoryTheory.CategoryStruct.comp t₂.inl …
    -/
  · intro m hm₁ hm₂
    /-
      case create.mk.intro.refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X₁ X₂ X₃ Y₁ : C
      f₁ : Quiver.Hom X₁ X₂
      f₂ : Quiver.Hom X₂ X₃
      i₁ : Quiver.Hom X₁ Y₁
      t₁ : CategoryTheory.Limits.PushoutCocone i₁ f₁
      i₂ : Quiver.Hom X₂ t₁.pt
      t₂ : CategoryTheory.Limits.PushoutCocone «i₂» f₂
      hi₂ : Eq «i₂» t₁.inr
      H : CategoryTheory.Limits.IsColimit t₁
      H' : CategoryTheory.Limits.IsColimit (t₁.pasteHoriz t₂ hi₂)
      s : CategoryTheory.Limits.PushoutCocone «i₂» f₂
      l : Quiver.Hom (t₁.pasteHoriz t₂ hi₂).pt s.pt
      hl : Eq (CategoryTheory.CategoryStruct.comp (t₁.pasteHoriz t₂ hi₂).inl l) (Cat …
      hl' : Eq (CategoryTheory.CategoryStruct.comp (t₁.pasteHoriz t₂ hi₂).inr l) s.inr
      m : Quiver.Hom t₂.pt s.pt
      hm₁ : Eq (CategoryTheory.CategoryStruct.comp t₂.inl m) s.inl
      hm₂ : Eq (CategoryTheory.CategoryStruct.comp t₂.inr m) s.inr
      ⊢ Eq m l
    -/
    apply PushoutCocone.IsColimit.hom_ext H'
      /-
        case create.mk.intro.refine_2.h₀
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X₁ X₂ X₃ Y₁ : C
        f₁ : Quiver.Hom X₁ X₂
        f₂ : Quiver.Hom X₂ X₃
        i₁ : Quiver.Hom X₁ Y₁
        t₁ : CategoryTheory.Limits.PushoutCocone i₁ f₁
        i₂ : Quiver.Hom X₂ t₁.pt
        t₂ : CategoryTheory.Limits.PushoutCocone «i₂» f₂
        hi₂ : Eq «i₂» t₁.inr
        H : CategoryTheory.Limits.IsColimit t₁
        H' : CategoryTheory.Limits.IsColimit (t₁.pasteHoriz t₂ hi₂)
        s : CategoryTheory.Limits.PushoutCocone «i₂» f₂
        l : Quiver.Hom (t₁.pasteHoriz t₂ hi₂).pt s.pt
        hl : Eq (CategoryTheory.CategoryStruct.comp (t₁.pasteHoriz t₂ hi₂).inl l) (Cat …
        hl' : Eq (CategoryTheory.CategoryStruct.comp (t₁.pasteHoriz t₂ hi₂).inr l) s.inr
        m : Quiver.Hom t₂.pt s.pt
        hm₁ : Eq (CategoryTheory.CategoryStruct.comp t₂.inl m) s.inl
        hm₂ : Eq (CategoryTheory.CategoryStruct.comp t₂.inr m) s.inr
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (t₁.pasteHoriz t₂ hi₂).inl m) (Catego …
      -/
    · simpa [← hm₁] using hl.symm
      /-
        🎉 no goals
      -/
      /-
        case create.mk.intro.refine_2.h₁
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X₁ X₂ X₃ Y₁ : C
        f₁ : Quiver.Hom X₁ X₂
        f₂ : Quiver.Hom X₂ X₃
        i₁ : Quiver.Hom X₁ Y₁
        t₁ : CategoryTheory.Limits.PushoutCocone i₁ f₁
        i₂ : Quiver.Hom X₂ t₁.pt
        t₂ : CategoryTheory.Limits.PushoutCocone «i₂» f₂
        hi₂ : Eq «i₂» t₁.inr
        H : CategoryTheory.Limits.IsColimit t₁
        H' : CategoryTheory.Limits.IsColimit (t₁.pasteHoriz t₂ hi₂)
        s : CategoryTheory.Limits.PushoutCocone «i₂» f₂
        l : Quiver.Hom (t₁.pasteHoriz t₂ hi₂).pt s.pt
        hl : Eq (CategoryTheory.CategoryStruct.comp (t₁.pasteHoriz t₂ hi₂).inl l) (Cat …
        hl' : Eq (CategoryTheory.CategoryStruct.comp (t₁.pasteHoriz t₂ hi₂).inr l) s.inr
        m : Quiver.Hom t₂.pt s.pt
        hm₁ : Eq (CategoryTheory.CategoryStruct.comp t₂.inl m) s.inl
        hm₂ : Eq (CategoryTheory.CategoryStruct.comp t₂.inr m) s.inr
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (t₁.pasteHoriz t₂ hi₂).inr m) (Catego …
      -/
    · simpa [← hm₂] using hl'.symm
      /-
        🎉 no goals
      -/


/-- Given that the left square is a pushout, the pasted square is a pushout iff the right square is.
-/
def pasteHorizIsPushoutEquiv (H : IsColimit t₁) :
    IsColimit (t₁.pasteHoriz t₂ hi₂) ≃ IsColimit t₂ where
  toFun H' := rightSquareIsPushout t₂ _ H H'
  invFun H' := pasteHorizIsPushout _ H H'
  left_inv _ := Subsingleton.elim _ _
  right_inv _ := Subsingleton.elim _ _


/-- The `PullbackCone` obtained by pasting two `PullbackCone`'s vertically -/
abbrev PushoutCocone.pasteVert
    (t₁ : PushoutCocone g₂ i₃) {i₂ : Y₂ ⟶ t₁.pt} (t₂ : PushoutCocone g₁ i₂) (hi₂ : i₂ = t₁.inl) :
    PushoutCocone (g₂ ≫ g₁) i₃ :=
  PushoutCocone.mk t₂.inl (t₁.inr ≫ t₂.inr)
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          Y₃ Y₂ Y₁ X₃ : C
          g₂ : Quiver.Hom Y₃ Y₂
          g₁ : Quiver.Hom Y₂ Y₁
          i₃ : Quiver.Hom Y₃ X₃
          t₁✝ : CategoryTheory.Limits.PushoutCocone g₂ i₃
          i₂✝ : Quiver.Hom Y₂ t₁✝.pt
          t₂✝ : CategoryTheory.Limits.PushoutCocone g₁ i₂✝
          hi₂✝ : Eq i₂✝ t₁✝.inl
          t₁ : CategoryTheory.Limits.PushoutCocone g₂ i₃
          i₂ : Quiver.Hom Y₂ t₁.pt
          t₂ : CategoryTheory.Limits.PushoutCocone g₁ i₂
          hi₂ : Eq i₂ t₁.inl
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp g …
        -/
    (by rw [← reassoc_of% t₁.condition, Category.assoc, t₂.condition, ← hi₂])
        /-
          🎉 no goals
        -/


local notation "X₂" => t₁.pt

local notation "f₂" => t₁.inr

local notation "i₂" => t₁.inl

local notation "X₁" => t₂.pt

local notation "f₁" => t₂.inr

local notation "i₁" => t₂.inl


/-- Pasting two pushout cocones vertically is isomorphic to the pushout cocone obtained by flipping
them, pasting horizontally, and then flipping the result again. -/
def PushoutCocone.pasteVertFlip : (t₁.pasteVert t₂ hi₂).flip ≅ (t₁.flip.pasteHoriz t₂.flip hi₂) :=
                                     /-
                                       C : Type u
                                       inst✝ : CategoryTheory.Category.{v, u} C
                                       Y₃ Y₂ Y₁ X₃ : C
                                       g₂ : Quiver.Hom Y₃ Y₂
                                       g₁ : Quiver.Hom Y₂ Y₁
                                       i₃ : Quiver.Hom Y₃ X₃
                                       t₁ : CategoryTheory.Limits.PushoutCocone g₂ i₃
                                       i₂ : Quiver.Hom Y₂ t₁.pt
                                       t₂ : CategoryTheory.Limits.PushoutCocone g₁ «i₂»
                                       hi₂ : Eq «i₂» t₁.inl
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (t₁.pasteVert t₂ hi₂).flip.inl (Categ …
                                     -/
                                     /-
                                       🎉 no goals
                                     -/
  PushoutCocone.ext (Iso.refl _) (by simp) (by simp)
                                               /-
                                                 🎉 no goals
                                               -/


/-- Given
```
Y₃ - i₃ -> X₃
|          |
g₂         f₂
∨          ∨
Y₂ - i₂ -> X₂
|          |
g₁         f₁
∨          ∨
Y₁ - i₁ -> X₁
```
The big square is a pushout if both the small squares are.
-/
def pasteVertIsPushout (H₁ : IsColimit t₁) (H₂ : IsColimit t₂) :
    IsColimit (t₁.pasteVert t₂ hi₂) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    Y₃ Y₂ Y₁ X₃ : C
    g₂ : Quiver.Hom Y₃ Y₂
    g₁ : Quiver.Hom Y₂ Y₁
    i₃ : Quiver.Hom Y₃ X₃
    t₁ : CategoryTheory.Limits.PushoutCocone g₂ i₃
    i₂ : Quiver.Hom Y₂ t₁.pt
    t₂ : CategoryTheory.Limits.PushoutCocone g₁ «i₂»
    hi₂ : Eq «i₂» t₁.inl
    H₁ : CategoryTheory.Limits.IsColimit t₁
    H₂ : CategoryTheory.Limits.IsColimit t₂
    ⊢ CategoryTheory.Limits.IsColimit (t₁.pasteVert t₂ hi₂)
  -/
  apply PushoutCocone.isColimitOfFlip <| IsColimit.ofIsoColimit _ (t₁.pasteVertFlip t₂ hi₂).symm
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    Y₃ Y₂ Y₁ X₃ : C
    g₂ : Quiver.Hom Y₃ Y₂
    g₁ : Quiver.Hom Y₂ Y₁
    i₃ : Quiver.Hom Y₃ X₃
    t₁ : CategoryTheory.Limits.PushoutCocone g₂ i₃
    i₂ : Quiver.Hom Y₂ t₁.pt
    t₂ : CategoryTheory.Limits.PushoutCocone g₁ «i₂»
    hi₂ : Eq «i₂» t₁.inl
    H₁ : CategoryTheory.Limits.IsColimit t₁
    H₂ : CategoryTheory.Limits.IsColimit t₂
    ⊢ CategoryTheory.Limits.IsColimit (t₁.flip.pasteHoriz t₂.flip hi₂)
  -/
  exact pasteHorizIsPushout hi₂ (PushoutCocone.flipIsColimit H₁) (PushoutCocone.flipIsColimit H₂)
  /-
    🎉 no goals
  -/


/-- Given
```
Y₃ - i₃ -> X₃
|          |
g₂         f₂
∨          ∨
Y₂ - i₂ -> X₂
|          |
g₁         f₁
∨          ∨
Y₁ - i₁ -> X₁
```
The bottom square is a pushout if the top square and the big square are.
-/
def botSquareIsPushout (H₁ : IsColimit t₁) (H₂ : IsColimit (t₁.pasteVert t₂ hi₂)) : IsColimit t₂ :=
  PushoutCocone.isColimitOfFlip
    (rightSquareIsPushout _ hi₂ (PushoutCocone.flipIsColimit H₁) (PushoutCocone.flipIsColimit H₂))


/-- Given that the top square is a pushout, the pasted square is a pushout iff the bottom square is.
-/
def pasteVertIsPushoutEquiv (H : IsColimit t₁) :
    IsColimit (t₁.pasteVert t₂ hi₂) ≃ IsColimit t₂ where
  toFun H' := botSquareIsPushout t₂ _ H H'
  invFun H' := pasteVertIsPushout _ H H'
  left_inv _ := Subsingleton.elim _ _
  right_inv _ := Subsingleton.elim _ _


instance hasPullbackHorizPaste : HasPullback (f' ≫ f) g :=
  HasLimit.mk {
    cone := (pullback.cone f g).pasteHoriz (pullback.cone f' (pullback.fst f g)) rfl
    isLimit := pasteHorizIsPullback rfl (pullback.isLimit f g)
      (pullback.isLimit f' (pullback.fst f g))
  }


/-- The canonical isomorphism `W ×[X] (X ×[Z] Y) ≅ W ×[Z] Y` -/
noncomputable def pullbackRightPullbackFstIso :
    pullback f' (pullback.fst f g) ≅ pullback (f' ≫ f) g :=
  IsLimit.conePointUniqueUpToIso
    (pasteHorizIsPullback rfl (pullback.isLimit f g) (pullback.isLimit f' (pullback.fst f g)))
    (pullback.isLimit (f' ≫ f) g)


@[reassoc (attr := simp)]
theorem pullbackRightPullbackFstIso_hom_fst :
    (pullbackRightPullbackFstIso f g f').hom ≫ pullback.fst (f' ≫ f) g =
      pullback.fst f' (pullback.fst f g) :=
  IsLimit.conePointUniqueUpToIso_hom_comp _ _ WalkingCospan.left


@[reassoc (attr := simp)]
theorem pullbackRightPullbackFstIso_hom_snd :
    (pullbackRightPullbackFstIso f g f').hom ≫ pullback.snd _ _ =
      pullback.snd f' (pullback.fst f g) ≫ pullback.snd f g :=
  IsLimit.conePointUniqueUpToIso_hom_comp _ _ WalkingCospan.right


@[reassoc (attr := simp)]
theorem pullbackRightPullbackFstIso_inv_fst :
    (pullbackRightPullbackFstIso f g f').inv ≫ pullback.fst f' (pullback.fst f g) =
      pullback.fst (f' ≫ f) g :=
  IsLimit.conePointUniqueUpToIso_inv_comp _ _ WalkingCospan.left


@[reassoc (attr := simp)]
theorem pullbackRightPullbackFstIso_inv_snd_snd :
    (pullbackRightPullbackFstIso f g f').inv ≫ pullback.snd _ _ ≫ pullback.snd _ _ =
      pullback.snd _ _ :=
  IsLimit.conePointUniqueUpToIso_inv_comp _ _ WalkingCospan.right


@[reassoc (attr := simp)]
theorem pullbackRightPullbackFstIso_inv_snd_fst :
    (pullbackRightPullbackFstIso f g f').inv ≫ pullback.snd _ _ ≫ pullback.fst _ _ =
      pullback.fst _ _ ≫ f' := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    f' : Quiver.Hom W X
    inst✝¹ : CategoryTheory.Limits.HasPullback f g
    inst✝ : CategoryTheory.Limits.HasPullback f' (CategoryTheory.Limits.pullback.f …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackRightP …
  -/
  rw [← pullback.condition]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    f' : Quiver.Hom W X
    inst✝¹ : CategoryTheory.Limits.HasPullback f g
    inst✝ : CategoryTheory.Limits.HasPullback f' (CategoryTheory.Limits.pullback.f …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackRightP …
  -/
  exact pullbackRightPullbackFstIso_inv_fst_assoc f g f' _
  /-
    🎉 no goals
  -/


instance hasPullbackVertPaste : HasPullback f (g' ≫ g) :=
  HasLimit.mk {
    cone := (pullback.cone f g).pasteVert (pullback.cone (pullback.snd f g) g') rfl
    isLimit := pasteVertIsPullback rfl (pullback.isLimit f g)
      (pullback.isLimit (pullback.snd f g) g')
  }


/-- The canonical isomorphism `(X ×[Z] Y) ×[Y] W ≅ X ×[Z] W` -/
def pullbackLeftPullbackSndIso :
    pullback (pullback.snd f g) g' ≅ pullback f (g' ≫ g) :=
  IsLimit.conePointUniqueUpToIso
      (pasteVertIsPullback rfl (pullback.isLimit f g) (pullback.isLimit (pullback.snd f g) g'))
      (pullback.isLimit f (g' ≫ g))


@[reassoc (attr := simp)]
theorem pullbackLeftPullbackSndIso_hom_fst :
    (pullbackLeftPullbackSndIso f g g').hom ≫ pullback.fst _ _ =
      pullback.fst _ _ ≫ pullback.fst _ _ :=
  IsLimit.conePointUniqueUpToIso_hom_comp _ _ WalkingCospan.left


@[reassoc (attr := simp)]
theorem pullbackLeftPullbackSndIso_hom_snd :
    (pullbackLeftPullbackSndIso f g g').hom ≫ pullback.snd _ _ = pullback.snd _ _ :=
  IsLimit.conePointUniqueUpToIso_hom_comp _ _ WalkingCospan.right


@[reassoc (attr := simp)]
theorem pullbackLeftPullbackSndIso_inv_fst :
    (pullbackLeftPullbackSndIso f g g').inv ≫ pullback.fst _ _ ≫ pullback.fst _ _ =
      pullback.fst _ _ :=
  IsLimit.conePointUniqueUpToIso_inv_comp _ _ WalkingCospan.left


@[reassoc (attr := simp)]
theorem pullbackLeftPullbackSndIso_inv_snd_snd :
    (pullbackLeftPullbackSndIso f g g').inv ≫ pullback.snd _ _ = pullback.snd _ _ :=
  IsLimit.conePointUniqueUpToIso_inv_comp _ _ WalkingCospan.right


@[reassoc (attr := simp)]
theorem pullbackLeftPullbackSndIso_inv_fst_snd :
    (pullbackLeftPullbackSndIso f g g').inv ≫ pullback.fst _ _ ≫ pullback.snd _ _ =
      pullback.snd _ _ ≫ g' := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    g' : Quiver.Hom W Y
    inst✝¹ : CategoryTheory.Limits.HasPullback f g
    inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.snd  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackLeftPu …
  -/
  rw [pullback.condition]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    g' : Quiver.Hom W Y
    inst✝¹ : CategoryTheory.Limits.HasPullback f g
    inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.snd  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackLeftPu …
  -/
  exact pullbackLeftPullbackSndIso_inv_snd_snd_assoc f g g' g'
  /-
    🎉 no goals
  -/


instance : HasPushout f (g ≫ g') :=
  HasColimit.mk {
    cocone := (pushout.cocone f g).pasteHoriz (pushout.cocone (pushout.inr f g) g') rfl
    isColimit := pasteHorizIsPushout rfl (pushout.isColimit f g)
      (pushout.isColimit (pushout.inr f g) g')
  }


/-- The canonical isomorphism `(Y ⨿[X] Z) ⨿[Z] W ≅ Y ⨿[X] W` -/
noncomputable def pushoutLeftPushoutInrIso :
    pushout (pushout.inr f g) g' ≅ pushout f (g ≫ g') :=
  IsColimit.coconePointUniqueUpToIso
    (pasteHorizIsPushout rfl (pushout.isColimit f g) (pushout.isColimit (pushout.inr f g) g'))
    (pushout.isColimit f (g ≫ g'))


@[reassoc (attr := simp)]
theorem inl_pushoutLeftPushoutInrIso_inv :
    (pushout.inl f (g ≫ g')) ≫ (pushoutLeftPushoutInrIso f g g').inv =
      (pushout.inl f g) ≫ (pushout.inl (pushout.inr f g) g') :=
  IsColimit.comp_coconePointUniqueUpToIso_inv _ _ WalkingSpan.left


@[reassoc (attr := simp)]
theorem inr_pushoutLeftPushoutInrIso_hom :
    (pushout.inr (pushout.inr f g) g') ≫ (pushoutLeftPushoutInrIso f g g').hom =
      (pushout.inr f (g ≫ g')) :=
  IsColimit.comp_coconePointUniqueUpToIso_hom (pasteHorizIsPushout _ _ _) _ WalkingSpan.right


@[reassoc (attr := simp)]
theorem inr_pushoutLeftPushoutInrIso_inv :
    (pushout.inr f (g ≫ g')) ≫ (pushoutLeftPushoutInrIso f g g').inv =
      (pushout.inr (pushout.inr f g) g') :=
  IsColimit.comp_coconePointUniqueUpToIso_inv _ _ WalkingSpan.right


@[reassoc (attr := simp)]
theorem inl_inl_pushoutLeftPushoutInrIso_hom :
    (pushout.inl f g) ≫ (pushout.inl (pushout.inr f g) g') ≫
      (pushoutLeftPushoutInrIso f g g').hom = (pushout.inl f (g ≫ g')) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    g' : Quiver.Hom Z W
    inst✝¹ : CategoryTheory.Limits.HasPushout f g
    inst✝ : CategoryTheory.Limits.HasPushout (CategoryTheory.Limits.pushout.inr f  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl f  …
  -/
  rw [← Category.assoc]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    g' : Quiver.Hom Z W
    inst✝¹ : CategoryTheory.Limits.HasPushout f g
    inst✝ : CategoryTheory.Limits.HasPushout (CategoryTheory.Limits.pushout.inr f  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  apply IsColimit.comp_coconePointUniqueUpToIso_hom (pasteHorizIsPushout _ _ _) _ WalkingSpan.left
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem inr_inl_pushoutLeftPushoutInrIso_hom :
    pushout.inr f g ≫ pushout.inl (pushout.inr f g) g' ≫ (pushoutLeftPushoutInrIso f g g').hom =
      g' ≫ pushout.inr f (g ≫ g') := by
  rw [← Category.assoc, ← Iso.eq_comp_inv, Category.assoc, inr_pushoutLeftPushoutInrIso_inv,
    pushout.condition]


instance hasPushoutVertPaste : HasPushout (f ≫ f') g :=
  HasColimit.mk {
    cocone := (pushout.cocone f g).pasteVert (pushout.cocone f' (pushout.inl f g)) rfl
    isColimit := pasteVertIsPushout rfl (pushout.isColimit f g)
      (pushout.isColimit f' (pushout.inl f g))
  }


/-- The canonical isomorphism `W ⨿[Y] (Y ⨿[X] Z) ≅ W ⨿[X] Z` -/
noncomputable def pushoutRightPushoutInlIso :
    pushout f' (pushout.inl f g) ≅ pushout (f ≫ f') g :=
  IsColimit.coconePointUniqueUpToIso
    (pasteVertIsPushout rfl (pushout.isColimit f g) (pushout.isColimit f' (pushout.inl f g)))
    (pushout.isColimit (f ≫ f') g)


@[reassoc (attr := simp)]
theorem inl_pushoutRightPushoutInlIso_inv :
    pushout.inl _ _ ≫ (pushoutRightPushoutInlIso f g f').inv = pushout.inl _ _ :=
  IsColimit.comp_coconePointUniqueUpToIso_inv _ _ WalkingSpan.left


@[reassoc (attr := simp)]
theorem inr_inr_pushoutRightPushoutInlIso_hom :
    pushout.inr _ _ ≫ pushout.inr _ _ ≫ (pushoutRightPushoutInlIso f g f').hom =
      pushout.inr _ _ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    f' : Quiver.Hom Y W
    inst✝¹ : CategoryTheory.Limits.HasPushout f g
    inst✝ : CategoryTheory.Limits.HasPushout f' (CategoryTheory.Limits.pushout.inl …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inr f  …
  -/
  rw [← Category.assoc]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    f' : Quiver.Hom Y W
    inst✝¹ : CategoryTheory.Limits.HasPushout f g
    inst✝ : CategoryTheory.Limits.HasPushout f' (CategoryTheory.Limits.pushout.inl …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  apply IsColimit.comp_coconePointUniqueUpToIso_hom (pasteVertIsPushout rfl _ _) _ WalkingSpan.right
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem inr_pushoutRightPushoutInlIso_inv :
    pushout.inr _ _ ≫ (pushoutRightPushoutInlIso f g f').inv =
      pushout.inr _ _ ≫ pushout.inr _ _ :=
  IsColimit.comp_coconePointUniqueUpToIso_inv _ _ WalkingSpan.right


@[reassoc (attr := simp)]
theorem inl_pushoutRightPushoutInlIso_hom :
    pushout.inl _ _ ≫ (pushoutRightPushoutInlIso f g f').hom = pushout.inl _ _ :=
  IsColimit.comp_coconePointUniqueUpToIso_hom (pasteVertIsPushout rfl _ _) _ WalkingSpan.left


@[reassoc (attr := simp)]
theorem inr_inl_pushoutRightPushoutInlIso_hom :
    pushout.inl _ _ ≫ pushout.inr _ _ ≫ (pushoutRightPushoutInlIso f g f').hom =
      f' ≫ pushout.inl _ _ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    f' : Quiver.Hom Y W
    inst✝¹ : CategoryTheory.Limits.HasPushout f g
    inst✝ : CategoryTheory.Limits.HasPushout f' (CategoryTheory.Limits.pushout.inl …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl f  …
  -/
  rw [← Category.assoc, ← pushout.condition, Category.assoc, inl_pushoutRightPushoutInlIso_hom]
  /-
    🎉 no goals
  -/


