/-- The equivalence `Sieve Y ≃ Sieve Y.left` for all `Y : Over X`. -/
def overEquiv {X : C} (Y : Over X) :
    Sieve Y ≃ Sieve Y.left where
  toFun S := Sieve.functorPushforward (Over.forget X) S
  invFun S' := Sieve.functorPullback (Over.forget X) S'
  left_inv S := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      Y : CategoryTheory.Over X
      S : CategoryTheory.Sieve Y
      ⊢ Eq ((fun S' => CategoryTheory.Sieve.functorPullback (CategoryTheory.Over.for …
    -/
    ext Z g
    /-
      case h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      Y : CategoryTheory.Over X
      S : CategoryTheory.Sieve Y
      Z : CategoryTheory.Over X
      g : Quiver.Hom Z Y
      ⊢ Iff (((fun S' => CategoryTheory.Sieve.functorPullback (CategoryTheory.Over.f …
    -/
    dsimp [Presieve.functorPullback, Presieve.functorPushforward]
    /-
      case h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      Y : CategoryTheory.Over X
      S : CategoryTheory.Sieve Y
      Z : CategoryTheory.Over X
      g : Quiver.Hom Z Y
      ⊢ Iff (Exists fun Z_1 => Exists fun g_1 => Exists fun h => And (S.arrows g_1)  …
    -/
    constructor
      /-
        case h.mp
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : C
        Y : CategoryTheory.Over X
        S : CategoryTheory.Sieve Y
        Z : CategoryTheory.Over X
        g : Quiver.Hom Z Y
        ⊢ (Exists fun Z_1 => Exists fun g_1 => Exists fun h => And (S.arrows g_1) (Eq  …
      -/
    · rintro ⟨W, a, b, h, w⟩
      let c : Z ⟶ W := Over.homMk b
        (by rw [← Over.w g, w, assoc, Over.w a])
      /-
        case h.mp.intro.intro.intro.intro
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : C
        Y : CategoryTheory.Over X
        S : CategoryTheory.Sieve Y
        Z : CategoryTheory.Over X
        g : Quiver.Hom Z Y
        W : CategoryTheory.Over X
        a : Quiver.Hom W Y
        b : Quiver.Hom Z.left W.left
        h : S.arrows a
        w : Eq g.left (CategoryTheory.CategoryStruct.comp b a.left)
        c : Quiver.Hom Z W := CategoryTheory.Over.homMk b ⋯
        ⊢ S.arrows g
      -/
      rw [show g = c ≫ a by ext; exact w]
      /-
        case h.mp.intro.intro.intro.intro
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : C
        Y : CategoryTheory.Over X
        S : CategoryTheory.Sieve Y
        Z : CategoryTheory.Over X
        g : Quiver.Hom Z Y
        W : CategoryTheory.Over X
        a : Quiver.Hom W Y
        b : Quiver.Hom Z.left W.left
        h : S.arrows a
        w : Eq g.left (CategoryTheory.CategoryStruct.comp b a.left)
        c : Quiver.Hom Z W := CategoryTheory.Over.homMk b ⋯
        ⊢ S.arrows (CategoryTheory.CategoryStruct.comp c a)
      -/
      exact S.downward_closed h _
      /-
        🎉 no goals
      -/
      /-
        case h.mpr
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : C
        Y : CategoryTheory.Over X
        S : CategoryTheory.Sieve Y
        Z : CategoryTheory.Over X
        g : Quiver.Hom Z Y
        ⊢ S.arrows g → Exists fun Z_1 => Exists fun g_1 => Exists fun h => And (S.arro …
      -/
    · intro h
      /-
        case h.mpr
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : C
        Y : CategoryTheory.Over X
        S : CategoryTheory.Sieve Y
        Z : CategoryTheory.Over X
        g : Quiver.Hom Z Y
        h : S.arrows g
        ⊢ Exists fun Z_1 => Exists fun g_1 => Exists fun h => And (S.arrows g_1) (Eq g …
      -/
      exact ⟨Z, g, 𝟙 _, h, by simp⟩
      /-
        🎉 no goals
      -/
  right_inv S := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      Y : CategoryTheory.Over X
      S : CategoryTheory.Sieve Y.left
      ⊢ Eq ((fun S => CategoryTheory.Sieve.functorPushforward (CategoryTheory.Over.f …
    -/
    ext Z g
    /-
      case h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      Y : CategoryTheory.Over X
      S : CategoryTheory.Sieve Y.left
      Z : C
      g : Quiver.Hom Z ((CategoryTheory.Over.forget X).obj Y)
      ⊢ Iff (((fun S => CategoryTheory.Sieve.functorPushforward (CategoryTheory.Over …
    -/
    dsimp [Presieve.functorPullback, Presieve.functorPushforward]
    /-
      case h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      Y : CategoryTheory.Over X
      S : CategoryTheory.Sieve Y.left
      Z : C
      g : Quiver.Hom Z ((CategoryTheory.Over.forget X).obj Y)
      ⊢ Iff (Exists fun Z_1 => Exists fun g_1 => Exists fun h => And (S.arrows g_1.l …
    -/
    constructor
      /-
        case h.mp
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : C
        Y : CategoryTheory.Over X
        S : CategoryTheory.Sieve Y.left
        Z : C
        g : Quiver.Hom Z ((CategoryTheory.Over.forget X).obj Y)
        ⊢ (Exists fun Z_1 => Exists fun g_1 => Exists fun h => And (S.arrows g_1.left) …
      -/
    · rintro ⟨W, a, b, h, rfl⟩
      /-
        case h.mp.intro.intro.intro.intro
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : C
        Y : CategoryTheory.Over X
        S : CategoryTheory.Sieve Y.left
        Z : C
        W : CategoryTheory.Over X
        a : Quiver.Hom W Y
        b : Quiver.Hom Z W.left
        h : S.arrows a.left
        ⊢ S.arrows (CategoryTheory.CategoryStruct.comp b a.left)
      -/
      exact S.downward_closed h _
      /-
        🎉 no goals
      -/
      /-
        case h.mpr
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : C
        Y : CategoryTheory.Over X
        S : CategoryTheory.Sieve Y.left
        Z : C
        g : Quiver.Hom Z ((CategoryTheory.Over.forget X).obj Y)
        ⊢ S.arrows g → Exists fun Z_1 => Exists fun g_1 => Exists fun h => And (S.arro …
      -/
    · intro h
      /-
        case h.mpr
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : C
        Y : CategoryTheory.Over X
        S : CategoryTheory.Sieve Y.left
        Z : C
        g : Quiver.Hom Z ((CategoryTheory.Over.forget X).obj Y)
        h : S.arrows g
        ⊢ Exists fun Z_1 => Exists fun g_1 => Exists fun h => And (S.arrows g_1.left)  …
      -/
      exact ⟨Over.mk ((g ≫ Y.hom)), Over.homMk g, 𝟙 _, h, by simp⟩
      /-
        🎉 no goals
      -/


@[simp]
lemma overEquiv_top {X : C} (Y : Over X) :
    overEquiv Y ⊤ = ⊤ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    Y : CategoryTheory.Over X
    ⊢ Eq ((CategoryTheory.Sieve.overEquiv Y) Top.top) Top.top
  -/
  ext Z g
  /-
    case h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    Y : CategoryTheory.Over X
    Z : C
    g : Quiver.Hom Z Y.left
    ⊢ Iff (((CategoryTheory.Sieve.overEquiv Y) Top.top).arrows g) (Top.top.arrows g)
  -/
  simp only [top_apply, iff_true]
  /-
    case h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    Y : CategoryTheory.Over X
    Z : C
    g : Quiver.Hom Z Y.left
    ⊢ ((CategoryTheory.Sieve.overEquiv Y) Top.top).arrows g
  -/
  dsimp [overEquiv, Presieve.functorPushforward]
  /-
    case h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    Y : CategoryTheory.Over X
    Z : C
    g : Quiver.Hom Z Y.left
    ⊢ Exists fun Z_1 => Exists fun g_1 => Exists fun h => And (Top.top.arrows g_1) …
  -/
  exact ⟨Y, 𝟙 Y, g, by simp, by simp⟩
  /-
    🎉 no goals
  -/


@[simp]
lemma overEquiv_symm_top {X : C} (Y : Over X) :
    (overEquiv Y).symm ⊤ = ⊤ :=
                              /-
                                C : Type u
                                inst✝ : CategoryTheory.Category.{v, u} C
                                X : C
                                Y : CategoryTheory.Over X
                                ⊢ Eq ((CategoryTheory.Sieve.overEquiv Y) ((CategoryTheory.Sieve.overEquiv Y).s …
                              -/
  (overEquiv Y).injective (by simp)
                              /-
                                🎉 no goals
                              -/


lemma overEquiv_le_overEquiv_iff {X : C} {Y : Over X} (R₁ R₂ : Sieve Y) :
    R₁.overEquiv Y ≤ R₂.overEquiv Y ↔ R₁ ≤ R₂ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    Y : CategoryTheory.Over X
    R₁ R₂ : CategoryTheory.Sieve Y
    ⊢ Iff (LE.le ((CategoryTheory.Sieve.overEquiv Y) R₁) ((CategoryTheory.Sieve.ov …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ Sieve.functorPushforward_monotone _ _ h⟩
  replace h : (overEquiv Y).symm (R₁.overEquiv Y) ≤ (overEquiv Y).symm (R₂.overEquiv Y) :=
    Sieve.functorPullback_monotone _ _ h
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    Y : CategoryTheory.Over X
    R₁ R₂ : CategoryTheory.Sieve Y
    h : LE.le ((CategoryTheory.Sieve.overEquiv Y).symm ((CategoryTheory.Sieve.over …
    ⊢ LE.le R₁ R₂
  -/
  simpa using h
  /-
    🎉 no goals
  -/


lemma overEquiv_pullback {X : C} {Y₁ Y₂ : Over X} (f : Y₁ ⟶ Y₂) (S : Sieve Y₂) :
    overEquiv _ (S.pullback f) = (overEquiv _ S).pullback f.left := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    Y₁ Y₂ : CategoryTheory.Over X
    f : Quiver.Hom Y₁ Y₂
    S : CategoryTheory.Sieve Y₂
    ⊢ Eq ((CategoryTheory.Sieve.overEquiv Y₁) (CategoryTheory.Sieve.pullback f S)) …
  -/
  ext Z g
  /-
    case h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    Y₁ Y₂ : CategoryTheory.Over X
    f : Quiver.Hom Y₁ Y₂
    S : CategoryTheory.Sieve Y₂
    Z : C
    g : Quiver.Hom Z Y₁.left
    ⊢ Iff (((CategoryTheory.Sieve.overEquiv Y₁) (CategoryTheory.Sieve.pullback f S …
  -/
  dsimp [overEquiv, Presieve.functorPushforward]
  /-
    case h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    Y₁ Y₂ : CategoryTheory.Over X
    f : Quiver.Hom Y₁ Y₂
    S : CategoryTheory.Sieve Y₂
    Z : C
    g : Quiver.Hom Z Y₁.left
    ⊢ Iff (Exists fun Z_1 => Exists fun g_1 => Exists fun h => And (S.arrows (Cate …
  -/
  constructor
    /-
      case h.mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      Y₁ Y₂ : CategoryTheory.Over X
      f : Quiver.Hom Y₁ Y₂
      S : CategoryTheory.Sieve Y₂
      Z : C
      g : Quiver.Hom Z Y₁.left
      ⊢ (Exists fun Z_1 => Exists fun g_1 => Exists fun h => And (S.arrows (Category …
    -/
  · rintro ⟨W, a, b, h, rfl⟩
    /-
      case h.mp.intro.intro.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      Y₁ Y₂ : CategoryTheory.Over X
      f : Quiver.Hom Y₁ Y₂
      S : CategoryTheory.Sieve Y₂
      Z : C
      W : CategoryTheory.Over X
      a : Quiver.Hom W Y₁
      b : Quiver.Hom Z W.left
      h : S.arrows (CategoryTheory.CategoryStruct.comp a f)
      ⊢ Exists fun Z_1 => Exists fun g => Exists fun h => And (S.arrows g) (Eq (Cate …
    -/
    exact ⟨W, a ≫ f, b, h, by simp⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      Y₁ Y₂ : CategoryTheory.Over X
      f : Quiver.Hom Y₁ Y₂
      S : CategoryTheory.Sieve Y₂
      Z : C
      g : Quiver.Hom Z Y₁.left
      ⊢ (Exists fun Z_1 => Exists fun g_1 => Exists fun h => And (S.arrows g_1) (Eq  …
    -/
  · rintro ⟨W, a, b, h, w⟩
    /-
      case h.mpr.intro.intro.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      Y₁ Y₂ : CategoryTheory.Over X
      f : Quiver.Hom Y₁ Y₂
      S : CategoryTheory.Sieve Y₂
      Z : C
      g : Quiver.Hom Z Y₁.left
      W : CategoryTheory.Over X
      a : Quiver.Hom W Y₂
      b : Quiver.Hom Z W.left
      h : S.arrows a
      w : Eq (CategoryTheory.CategoryStruct.comp g f.left) (CategoryTheory.CategoryS …
      ⊢ Exists fun Z_1 => Exists fun g_1 => Exists fun h => And (S.arrows (CategoryT …
    -/
    let T := Over.mk (b ≫ W.hom)
    /-
      case h.mpr.intro.intro.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      Y₁ Y₂ : CategoryTheory.Over X
      f : Quiver.Hom Y₁ Y₂
      S : CategoryTheory.Sieve Y₂
      Z : C
      g : Quiver.Hom Z Y₁.left
      W : CategoryTheory.Over X
      a : Quiver.Hom W Y₂
      b : Quiver.Hom Z W.left
      h : S.arrows a
      w : Eq (CategoryTheory.CategoryStruct.comp g f.left) (CategoryTheory.CategoryS …
      T : CategoryTheory.Over ((CategoryTheory.Functor.fromPUnit X).obj W.right) :=  …
      ⊢ Exists fun Z_1 => Exists fun g_1 => Exists fun h => And (S.arrows (CategoryT …
    -/
    let c : T ⟶ Y₁ := Over.homMk g (by dsimp [T]; rw [← Over.w a, ← reassoc_of% w, Over.w f])
    /-
      case h.mpr.intro.intro.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      Y₁ Y₂ : CategoryTheory.Over X
      f : Quiver.Hom Y₁ Y₂
      S : CategoryTheory.Sieve Y₂
      Z : C
      g : Quiver.Hom Z Y₁.left
      W : CategoryTheory.Over X
      a : Quiver.Hom W Y₂
      b : Quiver.Hom Z W.left
      h : S.arrows a
      w : Eq (CategoryTheory.CategoryStruct.comp g f.left) (CategoryTheory.CategoryS …
      T : CategoryTheory.Over ((CategoryTheory.Functor.fromPUnit X).obj W.right) :=  …
      c : Quiver.Hom T Y₁ := CategoryTheory.Over.homMk g ⋯
      ⊢ Exists fun Z_1 => Exists fun g_1 => Exists fun h => And (S.arrows (CategoryT …
    -/
    let d : T ⟶ W := Over.homMk b
    /-
      case h.mpr.intro.intro.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      Y₁ Y₂ : CategoryTheory.Over X
      f : Quiver.Hom Y₁ Y₂
      S : CategoryTheory.Sieve Y₂
      Z : C
      g : Quiver.Hom Z Y₁.left
      W : CategoryTheory.Over X
      a : Quiver.Hom W Y₂
      b : Quiver.Hom Z W.left
      h : S.arrows a
      w : Eq (CategoryTheory.CategoryStruct.comp g f.left) (CategoryTheory.CategoryS …
      T : CategoryTheory.Over ((CategoryTheory.Functor.fromPUnit X).obj W.right) :=  …
      c : Quiver.Hom T Y₁ := CategoryTheory.Over.homMk g ⋯
      d : Quiver.Hom T W := CategoryTheory.Over.homMk b ⋯
      ⊢ Exists fun Z_1 => Exists fun g_1 => Exists fun h => And (S.arrows (CategoryT …
    -/
    refine ⟨T, c, 𝟙 Z, ?_, by simp [T, c]⟩
    /-
      case h.mpr.intro.intro.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      Y₁ Y₂ : CategoryTheory.Over X
      f : Quiver.Hom Y₁ Y₂
      S : CategoryTheory.Sieve Y₂
      Z : C
      g : Quiver.Hom Z Y₁.left
      W : CategoryTheory.Over X
      a : Quiver.Hom W Y₂
      b : Quiver.Hom Z W.left
      h : S.arrows a
      w : Eq (CategoryTheory.CategoryStruct.comp g f.left) (CategoryTheory.CategoryS …
      T : CategoryTheory.Over ((CategoryTheory.Functor.fromPUnit X).obj W.right) :=  …
      c : Quiver.Hom T Y₁ := CategoryTheory.Over.homMk g ⋯
      d : Quiver.Hom T W := CategoryTheory.Over.homMk b ⋯
      ⊢ S.arrows (CategoryTheory.CategoryStruct.comp c f)
    -/
    rw [show c ≫ f = d ≫ a by ext; exact w]
    /-
      case h.mpr.intro.intro.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : C
      Y₁ Y₂ : CategoryTheory.Over X
      f : Quiver.Hom Y₁ Y₂
      S : CategoryTheory.Sieve Y₂
      Z : C
      g : Quiver.Hom Z Y₁.left
      W : CategoryTheory.Over X
      a : Quiver.Hom W Y₂
      b : Quiver.Hom Z W.left
      h : S.arrows a
      w : Eq (CategoryTheory.CategoryStruct.comp g f.left) (CategoryTheory.CategoryS …
      T : CategoryTheory.Over ((CategoryTheory.Functor.fromPUnit X).obj W.right) :=  …
      c : Quiver.Hom T Y₁ := CategoryTheory.Over.homMk g ⋯
      d : Quiver.Hom T W := CategoryTheory.Over.homMk b ⋯
      ⊢ S.arrows (CategoryTheory.CategoryStruct.comp d a)
    -/
    exact S.downward_closed h _
    /-
      🎉 no goals
    -/


@[simp]
lemma overEquiv_symm_iff {X : C} {Y : Over X} (S : Sieve Y.left) {Z : Over X} (f : Z ⟶ Y) :
    (overEquiv Y).symm S f ↔ S f.left := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    Y : CategoryTheory.Over X
    S : CategoryTheory.Sieve Y.left
    Z : CategoryTheory.Over X
    f : Quiver.Hom Z Y
    ⊢ Iff (((CategoryTheory.Sieve.overEquiv Y).symm S).arrows f) (S.arrows f.left)
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma overEquiv_iff {X : C} {Y : Over X} (S : Sieve Y) {Z : C} (f : Z ⟶ Y.left) :
                         /-
                           C : Type u
                           inst✝ : CategoryTheory.Category.{v, u} C
                           X : C
                           Y : CategoryTheory.Over X
                           S : CategoryTheory.Sieve Y
                           Z : C
                           f : Quiver.Hom Z Y.left
                           ⊢ Eq (CategoryTheory.CategoryStruct.comp f Y.hom) (CategoryTheory.Over.mk (Cat …
                         -/
    overEquiv Y S f ↔ S (Over.homMk f : Over.mk (f ≫ Y.hom) ⟶ Y) := by
                         /-
                           🎉 no goals
                         -/
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    Y : CategoryTheory.Over X
    S : CategoryTheory.Sieve Y
    Z : C
    f : Quiver.Hom Z Y.left
    ⊢ Iff (((CategoryTheory.Sieve.overEquiv Y) S).arrows f) (S.arrows (CategoryThe …
  -/
  obtain ⟨S, rfl⟩ := (overEquiv Y).symm.surjective S
  /-
    case intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    Y : CategoryTheory.Over X
    Z : C
    f : Quiver.Hom Z Y.left
    S : CategoryTheory.Sieve Y.left
    ⊢ Iff (((CategoryTheory.Sieve.overEquiv Y) ((CategoryTheory.Sieve.overEquiv Y) …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma functorPushforward_over_map {X Y : C} (f : X ⟶ Y) (Z : Over X) (S : Sieve Z.left) :
    Sieve.functorPushforward (Over.map f) ((Sieve.overEquiv Z).symm S) =
      (Sieve.overEquiv ((Over.map f).obj Z)).symm S := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    Z : CategoryTheory.Over X
    S : CategoryTheory.Sieve Z.left
    ⊢ Eq (CategoryTheory.Sieve.functorPushforward (CategoryTheory.Over.map f) ((Ca …
  -/
  ext W g
  /-
    case h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    Z : CategoryTheory.Over X
    S : CategoryTheory.Sieve Z.left
    W : CategoryTheory.Over Y
    g : Quiver.Hom W ((CategoryTheory.Over.map f).obj Z)
    ⊢ Iff ((CategoryTheory.Sieve.functorPushforward (CategoryTheory.Over.map f) (( …
  -/
  constructor
    /-
      case h.mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      f : Quiver.Hom X Y
      Z : CategoryTheory.Over X
      S : CategoryTheory.Sieve Z.left
      W : CategoryTheory.Over Y
      g : Quiver.Hom W ((CategoryTheory.Over.map f).obj Z)
      ⊢ (CategoryTheory.Sieve.functorPushforward (CategoryTheory.Over.map f) ((Categ …
    -/
  · rintro ⟨T, a, b, ha, rfl⟩
    /-
      case h.mp.intro.intro.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      f : Quiver.Hom X Y
      Z : CategoryTheory.Over X
      S : CategoryTheory.Sieve Z.left
      W : CategoryTheory.Over Y
      T : CategoryTheory.Over X
      a : Quiver.Hom T Z
      b : Quiver.Hom W ((CategoryTheory.Over.map f).obj T)
      ha : ((CategoryTheory.Sieve.overEquiv Z).symm S).arrows a
      ⊢ ((CategoryTheory.Sieve.overEquiv ((CategoryTheory.Over.map f).obj Z)).symm S …
    -/
    exact S.downward_closed ha _
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      f : Quiver.Hom X Y
      Z : CategoryTheory.Over X
      S : CategoryTheory.Sieve Z.left
      W : CategoryTheory.Over Y
      g : Quiver.Hom W ((CategoryTheory.Over.map f).obj Z)
      ⊢ ((CategoryTheory.Sieve.overEquiv ((CategoryTheory.Over.map f).obj Z)).symm S …
    -/
  · intro hg
    exact ⟨Over.mk (g.left ≫ Z.hom), Over.homMk g.left,
      Over.homMk (𝟙 _) (by simpa using Over.w g), hg, by aesop_cat⟩


/-- The Grothendieck topology on the category `Over X` for any `X : C` that is
induced by a Grothendieck topology on `C`. -/
def over (X : C) : GrothendieckTopology (Over X) where
  sieves Y S := Sieve.overEquiv Y S ∈ J Y.left
  top_mem' Y := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      X : C
      Y : CategoryTheory.Over X
      ⊢ Membership.mem ((fun Y S => Membership.mem (J Y.left) ((CategoryTheory.Sieve …
    -/
    change _ ∈ J Y.left
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      X : C
      Y : CategoryTheory.Over X
      ⊢ Membership.mem (J Y.left) ((CategoryTheory.Sieve.overEquiv Y) Top.top)
    -/
    simp
    /-
      🎉 no goals
    -/
  pullback_stable' Y₁ Y₂ S₁ f h₁ := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      X : C
      Y₁ Y₂ : CategoryTheory.Over X
      S₁ : CategoryTheory.Sieve Y₁
      f : Quiver.Hom Y₂ Y₁
      h₁ : Membership.mem ((fun Y S => Membership.mem (J Y.left) ((CategoryTheory.Si …
      ⊢ Membership.mem ((fun Y S => Membership.mem (J Y.left) ((CategoryTheory.Sieve …
    -/
    change _ ∈ J _ at h₁ ⊢
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      X : C
      Y₁ Y₂ : CategoryTheory.Over X
      S₁ : CategoryTheory.Sieve Y₁
      f : Quiver.Hom Y₂ Y₁
      h₁ : Membership.mem (J Y₁.left) ((CategoryTheory.Sieve.overEquiv Y₁) S₁)
      ⊢ Membership.mem (J Y₂.left) ((CategoryTheory.Sieve.overEquiv Y₂) (CategoryThe …
    -/
    rw [Sieve.overEquiv_pullback]
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      X : C
      Y₁ Y₂ : CategoryTheory.Over X
      S₁ : CategoryTheory.Sieve Y₁
      f : Quiver.Hom Y₂ Y₁
      h₁ : Membership.mem (J Y₁.left) ((CategoryTheory.Sieve.overEquiv Y₁) S₁)
      ⊢ Membership.mem (J Y₂.left) (CategoryTheory.Sieve.pullback f.left ((CategoryT …
    -/
    exact J.pullback_stable _ h₁
    /-
      🎉 no goals
    -/
  transitive' Y S (hS : _ ∈ J _) R hR := J.transitive hS _ (fun Z f hf => by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      X : C
      Y : CategoryTheory.Over X
      S : CategoryTheory.Sieve Y
      hS : Membership.mem (J Y.left) ((CategoryTheory.Sieve.overEquiv Y) S)
      R : CategoryTheory.Sieve Y
      hR : ∀ ⦃Y_1 : CategoryTheory.Over X⦄ ⦃f : Quiver.Hom Y_1 Y⦄, S.arrows f → Memb …
      Z : C
      f : Quiver.Hom Z Y.left
      hf : ((CategoryTheory.Sieve.overEquiv Y) S).arrows f
      ⊢ Membership.mem (J Z) (CategoryTheory.Sieve.pullback f ((CategoryTheory.Sieve …
    -/
    have hf' : _ ∈ J _ := hR ((Sieve.overEquiv_iff _ _).1 hf)
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      X : C
      Y : CategoryTheory.Over X
      S : CategoryTheory.Sieve Y
      hS : Membership.mem (J Y.left) ((CategoryTheory.Sieve.overEquiv Y) S)
      R : CategoryTheory.Sieve Y
      hR : ∀ ⦃Y_1 : CategoryTheory.Over X⦄ ⦃f : Quiver.Hom Y_1 Y⦄, S.arrows f → Memb …
      Z : C
      f : Quiver.Hom Z Y.left
      hf : ((CategoryTheory.Sieve.overEquiv Y) S).arrows f
      hf' : Membership.mem (J (CategoryTheory.Over.mk (CategoryTheory.CategoryStruct …
      ⊢ Membership.mem (J Z) (CategoryTheory.Sieve.pullback f ((CategoryTheory.Sieve …
    -/
    rw [Sieve.overEquiv_pullback] at hf'
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      X : C
      Y : CategoryTheory.Over X
      S : CategoryTheory.Sieve Y
      hS : Membership.mem (J Y.left) ((CategoryTheory.Sieve.overEquiv Y) S)
      R : CategoryTheory.Sieve Y
      hR : ∀ ⦃Y_1 : CategoryTheory.Over X⦄ ⦃f : Quiver.Hom Y_1 Y⦄, S.arrows f → Memb …
      Z : C
      f : Quiver.Hom Z Y.left
      hf : ((CategoryTheory.Sieve.overEquiv Y) S).arrows f
      hf' : Membership.mem (J (CategoryTheory.Over.mk (CategoryTheory.CategoryStruct …
      ⊢ Membership.mem (J Z) (CategoryTheory.Sieve.pullback f ((CategoryTheory.Sieve …
    -/
    exact hf')
    /-
      🎉 no goals
    -/


lemma mem_over_iff {X : C} {Y : Over X} (S : Sieve Y) :
    S ∈ (J.over X) Y ↔ Sieve.overEquiv _ S ∈ J Y.left := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    X : C
    Y : CategoryTheory.Over X
    S : CategoryTheory.Sieve Y
    ⊢ Iff (Membership.mem ((J.over X) Y) S) (Membership.mem (J Y.left) ((CategoryT …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma overEquiv_symm_mem_over {X : C} (Y : Over X) (S : Sieve Y.left) (hS : S ∈ J Y.left) :
    (Sieve.overEquiv Y).symm S ∈ (J.over X) Y := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    X : C
    Y : CategoryTheory.Over X
    S : CategoryTheory.Sieve Y.left
    hS : Membership.mem (J Y.left) S
    ⊢ Membership.mem ((J.over X) Y) ((CategoryTheory.Sieve.overEquiv Y).symm S)
  -/
  simpa only [mem_over_iff, Equiv.apply_symm_apply] using hS
  /-
    🎉 no goals
  -/


lemma over_forget_coverPreserving (X : C) :
    CoverPreserving (J.over X) J (Over.forget X) where
  cover_preserve hS := hS


lemma over_forget_compatiblePreserving (X : C) :
    CompatiblePreserving J (Over.forget X) where
  compatible {_ Z _ _ hx Y₁ Y₂ W f₁ f₂ g₁ g₂ hg₁ hg₂ h} := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      X : C
      x✝² : CategoryTheory.Sheaf J (Type u_1)
      Z : CategoryTheory.Over X
      x✝¹ : CategoryTheory.Presieve Z
      x✝ : CategoryTheory.Presieve.FamilyOfElements ((CategoryTheory.Over.forget X). …
      hx : x✝.Compatible
      Y₁ Y₂ : CategoryTheory.Over X
      W : C
      f₁ : Quiver.Hom W ((CategoryTheory.Over.forget X).obj Y₁)
      f₂ : Quiver.Hom W ((CategoryTheory.Over.forget X).obj Y₂)
      g₁ : Quiver.Hom Y₁ Z
      g₂ : Quiver.Hom Y₂ Z
      hg₁ : x✝¹ g₁
      hg₂ : x✝¹ g₂
      h : Eq (CategoryTheory.CategoryStruct.comp f₁ ((CategoryTheory.Over.forget X). …
      ⊢ Eq (x✝².val.map f₁.op (x✝ g₁ hg₁)) (x✝².val.map f₂.op (x✝ g₂ hg₂))
    -/
    let W' : Over X := Over.mk (f₁ ≫ Y₁.hom)
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      X : C
      x✝² : CategoryTheory.Sheaf J (Type u_1)
      Z : CategoryTheory.Over X
      x✝¹ : CategoryTheory.Presieve Z
      x✝ : CategoryTheory.Presieve.FamilyOfElements ((CategoryTheory.Over.forget X). …
      hx : x✝.Compatible
      Y₁ Y₂ : CategoryTheory.Over X
      W : C
      f₁ : Quiver.Hom W ((CategoryTheory.Over.forget X).obj Y₁)
      f₂ : Quiver.Hom W ((CategoryTheory.Over.forget X).obj Y₂)
      g₁ : Quiver.Hom Y₁ Z
      g₂ : Quiver.Hom Y₂ Z
      hg₁ : x✝¹ g₁
      hg₂ : x✝¹ g₂
      h : Eq (CategoryTheory.CategoryStruct.comp f₁ ((CategoryTheory.Over.forget X). …
      W' : CategoryTheory.Over X := CategoryTheory.Over.mk (CategoryTheory.CategoryS …
      ⊢ Eq (x✝².val.map f₁.op (x✝ g₁ hg₁)) (x✝².val.map f₂.op (x✝ g₂ hg₂))
    -/
    let g₁' : W' ⟶ Y₁ := Over.homMk f₁
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      X : C
      x✝² : CategoryTheory.Sheaf J (Type u_1)
      Z : CategoryTheory.Over X
      x✝¹ : CategoryTheory.Presieve Z
      x✝ : CategoryTheory.Presieve.FamilyOfElements ((CategoryTheory.Over.forget X). …
      hx : x✝.Compatible
      Y₁ Y₂ : CategoryTheory.Over X
      W : C
      f₁ : Quiver.Hom W ((CategoryTheory.Over.forget X).obj Y₁)
      f₂ : Quiver.Hom W ((CategoryTheory.Over.forget X).obj Y₂)
      g₁ : Quiver.Hom Y₁ Z
      g₂ : Quiver.Hom Y₂ Z
      hg₁ : x✝¹ g₁
      hg₂ : x✝¹ g₂
      h : Eq (CategoryTheory.CategoryStruct.comp f₁ ((CategoryTheory.Over.forget X). …
      W' : CategoryTheory.Over X := CategoryTheory.Over.mk (CategoryTheory.CategoryS …
      g₁' : Quiver.Hom W' Y₁ := CategoryTheory.Over.homMk f₁ ⋯
      ⊢ Eq (x✝².val.map f₁.op (x✝ g₁ hg₁)) (x✝².val.map f₂.op (x✝ g₂ hg₂))
    -/
    let g₂' : W' ⟶ Y₂ := Over.homMk f₂ (by simpa using h.symm =≫ Z.hom)
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      X : C
      x✝² : CategoryTheory.Sheaf J (Type u_1)
      Z : CategoryTheory.Over X
      x✝¹ : CategoryTheory.Presieve Z
      x✝ : CategoryTheory.Presieve.FamilyOfElements ((CategoryTheory.Over.forget X). …
      hx : x✝.Compatible
      Y₁ Y₂ : CategoryTheory.Over X
      W : C
      f₁ : Quiver.Hom W ((CategoryTheory.Over.forget X).obj Y₁)
      f₂ : Quiver.Hom W ((CategoryTheory.Over.forget X).obj Y₂)
      g₁ : Quiver.Hom Y₁ Z
      g₂ : Quiver.Hom Y₂ Z
      hg₁ : x✝¹ g₁
      hg₂ : x✝¹ g₂
      h : Eq (CategoryTheory.CategoryStruct.comp f₁ ((CategoryTheory.Over.forget X). …
      W' : CategoryTheory.Over X := CategoryTheory.Over.mk (CategoryTheory.CategoryS …
      g₁' : Quiver.Hom W' Y₁ := CategoryTheory.Over.homMk f₁ ⋯
      g₂' : Quiver.Hom W' Y₂ := CategoryTheory.Over.homMk f₂ ⋯
      ⊢ Eq (x✝².val.map f₁.op (x✝ g₁ hg₁)) (x✝².val.map f₂.op (x✝ g₂ hg₂))
    -/
    exact hx g₁' g₂' hg₁ hg₂ (by ext; exact h)
    /-
      🎉 no goals
    -/


instance (X : C) : (Over.forget X).IsCocontinuous (J.over X) J where
  cover_lift hS := J.overEquiv_symm_mem_over _ _ hS


instance (X : C) : (Over.forget X).IsContinuous (J.over X) J :=
  Functor.isContinuous_of_coverPreserving
    (over_forget_compatiblePreserving J X)
    (over_forget_coverPreserving J X)


/-- The pullback functor `Sheaf J A ⥤ Sheaf (J.over X) A` -/
abbrev overPullback (A : Type u') [Category.{v'} A] (X : C) :
    Sheaf J A ⥤ Sheaf (J.over X) A :=
  (Over.forget X).sheafPushforwardContinuous _ _ _


lemma over_map_coverPreserving {X Y : C} (f : X ⟶ Y) :
    CoverPreserving (J.over X) (J.over Y) (Over.map f) where
  cover_preserve {U S} hS := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      X Y : C
      f : Quiver.Hom X Y
      U : CategoryTheory.Over X
      S : CategoryTheory.Sieve U
      hS : Membership.mem ((J.over X) U) S
      ⊢ Membership.mem ((J.over Y) ((CategoryTheory.Over.map f).obj U)) (CategoryThe …
    -/
    obtain ⟨S, rfl⟩ := (Sieve.overEquiv U).symm.surjective S
    /-
      case intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      X Y : C
      f : Quiver.Hom X Y
      U : CategoryTheory.Over X
      S : CategoryTheory.Sieve U.left
      hS : Membership.mem ((J.over X) U) ((CategoryTheory.Sieve.overEquiv U).symm S)
      ⊢ Membership.mem ((J.over Y) ((CategoryTheory.Over.map f).obj U)) (CategoryThe …
    -/
    rw [Sieve.functorPushforward_over_map]
    /-
      case intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      X Y : C
      f : Quiver.Hom X Y
      U : CategoryTheory.Over X
      S : CategoryTheory.Sieve U.left
      hS : Membership.mem ((J.over X) U) ((CategoryTheory.Sieve.overEquiv U).symm S)
      ⊢ Membership.mem ((J.over Y) ((CategoryTheory.Over.map f).obj U)) ((CategoryTh …
    -/
    apply overEquiv_symm_mem_over
    /-
      case intro.hS
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      X Y : C
      f : Quiver.Hom X Y
      U : CategoryTheory.Over X
      S : CategoryTheory.Sieve U.left
      hS : Membership.mem ((J.over X) U) ((CategoryTheory.Sieve.overEquiv U).symm S)
      ⊢ Membership.mem (J ((CategoryTheory.Over.map f).obj U).left) S
    -/
    simpa [mem_over_iff] using hS
    /-
      🎉 no goals
    -/


lemma over_map_compatiblePreserving {X Y : C} (f : X ⟶ Y) :
    CompatiblePreserving (J.over Y) (Over.map f) where
  compatible {F Z _ x hx Y₁ Y₂ W f₁ f₂ g₁ g₂ hg₁ hg₂ h} := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      X Y : C
      f : Quiver.Hom X Y
      F : CategoryTheory.Sheaf (J.over Y) (Type u_1)
      Z : CategoryTheory.Over X
      x✝ : CategoryTheory.Presieve Z
      x : CategoryTheory.Presieve.FamilyOfElements ((CategoryTheory.Over.map f).op.c …
      hx : x.Compatible
      Y₁ Y₂ : CategoryTheory.Over X
      W : CategoryTheory.Over Y
      f₁ : Quiver.Hom W ((CategoryTheory.Over.map f).obj Y₁)
      f₂ : Quiver.Hom W ((CategoryTheory.Over.map f).obj Y₂)
      g₁ : Quiver.Hom Y₁ Z
      g₂ : Quiver.Hom Y₂ Z
      hg₁ : x✝ g₁
      hg₂ : x✝ g₂
      h : Eq (CategoryTheory.CategoryStruct.comp f₁ ((CategoryTheory.Over.map f).map …
      ⊢ Eq (F.val.map f₁.op (x g₁ hg₁)) (F.val.map f₂.op (x g₂ hg₂))
    -/
    let W' : Over X := Over.mk (f₁.left ≫ Y₁.hom)
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      X Y : C
      f : Quiver.Hom X Y
      F : CategoryTheory.Sheaf (J.over Y) (Type u_1)
      Z : CategoryTheory.Over X
      x✝ : CategoryTheory.Presieve Z
      x : CategoryTheory.Presieve.FamilyOfElements ((CategoryTheory.Over.map f).op.c …
      hx : x.Compatible
      Y₁ Y₂ : CategoryTheory.Over X
      W : CategoryTheory.Over Y
      f₁ : Quiver.Hom W ((CategoryTheory.Over.map f).obj Y₁)
      f₂ : Quiver.Hom W ((CategoryTheory.Over.map f).obj Y₂)
      g₁ : Quiver.Hom Y₁ Z
      g₂ : Quiver.Hom Y₂ Z
      hg₁ : x✝ g₁
      hg₂ : x✝ g₂
      h : Eq (CategoryTheory.CategoryStruct.comp f₁ ((CategoryTheory.Over.map f).map …
      W' : CategoryTheory.Over X := CategoryTheory.Over.mk (CategoryTheory.CategoryS …
      ⊢ Eq (F.val.map f₁.op (x g₁ hg₁)) (F.val.map f₂.op (x g₂ hg₂))
    -/
    let g₁' : W' ⟶ Y₁ := Over.homMk f₁.left
    let g₂' : W' ⟶ Y₂ := Over.homMk f₂.left
      (by simpa using (Over.forget _).congr_map h.symm =≫ Z.hom)
    let e : (Over.map f).obj W' ≅ W := Over.isoMk (Iso.refl _)
      (by simpa [W'] using (Over.w f₁).symm)
    convert congr_arg (F.val.map e.inv.op)
      (hx g₁' g₂' hg₁ hg₂ (by ext; exact (Over.forget _).congr_map h)) using 1
    all_goals
      dsimp [e, W', g₁', g₂']
      rw [← FunctorToTypes.map_comp_apply]
      apply congr_fun
      congr 1
      rw [← op_comp]
      congr 1
      ext
      simp


instance {X Y : C} (f : X ⟶ Y) : (Over.map f).IsContinuous (J.over X) (J.over Y) :=
  Functor.isContinuous_of_coverPreserving
    (over_map_compatiblePreserving J f)
    (over_map_coverPreserving J f)


/-- The pullback functor `Sheaf (J.over Y) A ⥤ Sheaf (J.over X) A` induced
by a morphism `f : X ⟶ Y`. -/
abbrev overMapPullback (A : Type u') [Category.{v'} A] {X Y : C} (f : X ⟶ Y) :
    Sheaf (J.over Y) A ⥤ Sheaf (J.over X) A :=
  (Over.map f).sheafPushforwardContinuous _ _ _


/-- Given `F : Sheaf J A` and `X : C`, this is the pullback of `F` on `J.over X`. -/
abbrev Sheaf.over {A : Type u'} [Category.{v'} A] (F : Sheaf J A) (X : C) :
    Sheaf (J.over X) A := (J.overPullback A X).obj F


