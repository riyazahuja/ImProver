/-- The `Quiver` instance obtained by pushing arrows of `V` along the map `σ : V → W` -/
@[nolint unusedArguments]
def Push (_ : V → W) :=
  W


instance [h : Nonempty W] : Nonempty (Push σ) :=
  h


/-- The quiver structure obtained by pushing arrows of `V` along the map `σ : V → W` -/
inductive PushQuiver {V : Type u} [Quiver.{v} V] {W : Type u₂} (σ : V → W) : W → W → Type max u u₂ v
  | arrow {X Y : V} (f : X ⟶ Y) : PushQuiver σ (σ X) (σ Y)


instance : Quiver (Push σ) :=
  ⟨PushQuiver σ⟩


/-- The prefunctor induced by pushing arrows via `σ` -/
def of : V ⥤q Push σ where
  obj := σ
  map f := PushQuiver.arrow f


@[simp]
theorem of_obj : (of σ).obj = σ :=
  rfl


/-- Given a function `τ : W → W'` and a prefunctor `φ : V ⥤q W'`, one can extend `τ` to be
a prefunctor `W ⥤q W'` if `τ` and `σ` factorize `φ` at the level of objects, where `W` is given
the pushforward quiver structure `Push σ`. -/
noncomputable def lift : Push σ ⥤q W' where
  obj := τ
  map :=
    @PushQuiver.rec V _ W σ (fun X Y _ => τ X ⟶ τ Y) @fun X Y f => by
      /-
        V : Type u_1
        inst✝¹ : Quiver V
        W : Type u_2
        σ : V → W
        W' : Type u_3
        inst✝ : Quiver W'
        φ : Prefunctor V W'
        τ : W → W'
        h : ∀ (x : V), Eq (φ.obj x) (τ (σ x))
        X Y : V
        f : Quiver.Hom X Y
        ⊢ (fun X Y x => Quiver.Hom (τ X) (τ Y)) (σ X) (σ Y) (Quiver.PushQuiver.arrow f)
      -/
      dsimp only
      /-
        V : Type u_1
        inst✝¹ : Quiver V
        W : Type u_2
        σ : V → W
        W' : Type u_3
        inst✝ : Quiver W'
        φ : Prefunctor V W'
        τ : W → W'
        h : ∀ (x : V), Eq (φ.obj x) (τ (σ x))
        X Y : V
        f : Quiver.Hom X Y
        ⊢ Quiver.Hom (τ (σ X)) (τ (σ Y))
      -/
      rw [← h X, ← h Y]
      /-
        V : Type u_1
        inst✝¹ : Quiver V
        W : Type u_2
        σ : V → W
        W' : Type u_3
        inst✝ : Quiver W'
        φ : Prefunctor V W'
        τ : W → W'
        h : ∀ (x : V), Eq (φ.obj x) (τ (σ x))
        X Y : V
        f : Quiver.Hom X Y
        ⊢ Quiver.Hom (φ.obj X) (φ.obj Y)
      -/
      exact φ.map f
      /-
        🎉 no goals
      -/


theorem lift_obj : (lift σ φ τ h).obj = τ :=
  rfl


theorem lift_comp : (of σ ⋙q lift σ φ τ h) = φ := by
  /-
    V : Type u_1
    inst✝¹ : Quiver V
    W : Type u_2
    σ : V → W
    W' : Type u_3
    inst✝ : Quiver W'
    φ : Prefunctor V W'
    τ : W → W'
    h : ∀ (x : V), Eq (φ.obj x) (τ (σ x))
    ⊢ Eq ((Quiver.Push.of σ).comp (Quiver.Push.lift σ φ τ h)) φ
  -/
  fapply Prefunctor.ext
    /-
      case h_obj
      V : Type u_1
      inst✝¹ : Quiver V
      W : Type u_2
      σ : V → W
      W' : Type u_3
      inst✝ : Quiver W'
      φ : Prefunctor V W'
      τ : W → W'
      h : ∀ (x : V), Eq (φ.obj x) (τ (σ x))
      ⊢ ∀ (X : V), Eq (((Quiver.Push.of σ).comp (Quiver.Push.lift σ φ τ h)).obj X) ( …
    -/
  · rintro X
    /-
      case h_obj
      V : Type u_1
      inst✝¹ : Quiver V
      W : Type u_2
      σ : V → W
      W' : Type u_3
      inst✝ : Quiver W'
      φ : Prefunctor V W'
      τ : W → W'
      h : ∀ (x : V), Eq (φ.obj x) (τ (σ x))
      X : V
      ⊢ Eq (((Quiver.Push.of σ).comp (Quiver.Push.lift σ φ τ h)).obj X) (φ.obj X)
    -/
    simp only [Prefunctor.comp_obj]
    /-
      case h_obj
      V : Type u_1
      inst✝¹ : Quiver V
      W : Type u_2
      σ : V → W
      W' : Type u_3
      inst✝ : Quiver W'
      φ : Prefunctor V W'
      τ : W → W'
      h : ∀ (x : V), Eq (φ.obj x) (τ (σ x))
      X : V
      ⊢ Eq ((Quiver.Push.lift σ φ τ h).obj ((Quiver.Push.of σ).obj X)) (φ.obj X)
    -/
    apply Eq.symm
    /-
      case h_obj.h
      V : Type u_1
      inst✝¹ : Quiver V
      W : Type u_2
      σ : V → W
      W' : Type u_3
      inst✝ : Quiver W'
      φ : Prefunctor V W'
      τ : W → W'
      h : ∀ (x : V), Eq (φ.obj x) (τ (σ x))
      X : V
      ⊢ Eq (φ.obj X) ((Quiver.Push.lift σ φ τ h).obj ((Quiver.Push.of σ).obj X))
    -/
    exact h X
    /-
      🎉 no goals
    -/
    /-
      case h_map
      V : Type u_1
      inst✝¹ : Quiver V
      W : Type u_2
      σ : V → W
      W' : Type u_3
      inst✝ : Quiver W'
      φ : Prefunctor V W'
      τ : W → W'
      h : ∀ (x : V), Eq (φ.obj x) (τ (σ x))
      ⊢ ∀ (X Y : V) (f : Quiver.Hom X Y), Eq (((Quiver.Push.of σ).comp (Quiver.Push. …
    -/
  · rintro X Y f
    /-
      case h_map
      V : Type u_1
      inst✝¹ : Quiver V
      W : Type u_2
      σ : V → W
      W' : Type u_3
      inst✝ : Quiver W'
      φ : Prefunctor V W'
      τ : W → W'
      h : ∀ (x : V), Eq (φ.obj x) (τ (σ x))
      X Y : V
      f : Quiver.Hom X Y
      ⊢ Eq (((Quiver.Push.of σ).comp (Quiver.Push.lift σ φ τ h)).map f) (Eq.recOn ⋯  …
    -/
    simp only [Prefunctor.comp_map]
    /-
      case h_map
      V : Type u_1
      inst✝¹ : Quiver V
      W : Type u_2
      σ : V → W
      W' : Type u_3
      inst✝ : Quiver W'
      φ : Prefunctor V W'
      τ : W → W'
      h : ∀ (x : V), Eq (φ.obj x) (τ (σ x))
      X Y : V
      f : Quiver.Hom X Y
      ⊢ Eq ((Quiver.Push.lift σ φ τ h).map ((Quiver.Push.of σ).map f)) (Eq.rec (Eq.r …
    -/
    apply eq_of_heq
    /-
      case h_map.h
      V : Type u_1
      inst✝¹ : Quiver V
      W : Type u_2
      σ : V → W
      W' : Type u_3
      inst✝ : Quiver W'
      φ : Prefunctor V W'
      τ : W → W'
      h : ∀ (x : V), Eq (φ.obj x) (τ (σ x))
      X Y : V
      f : Quiver.Hom X Y
      ⊢ HEq ((Quiver.Push.lift σ φ τ h).map ((Quiver.Push.of σ).map f)) (Eq.rec (Eq. …
    -/
    iterate 2 apply (cast_heq _ _).trans
    /-
      case h_map.h
      V : Type u_1
      inst✝¹ : Quiver V
      W : Type u_2
      σ : V → W
      W' : Type u_3
      inst✝ : Quiver W'
      φ : Prefunctor V W'
      τ : W → W'
      h : ∀ (x : V), Eq (φ.obj x) (τ (σ x))
      X Y : V
      f : Quiver.Hom X Y
      ⊢ HEq (φ.map f) (Eq.rec (Eq.rec (φ.map f) ⋯) ⋯)
    -/
    apply HEq.symm
    /-
      case h_map.h.h
      V : Type u_1
      inst✝¹ : Quiver V
      W : Type u_2
      σ : V → W
      W' : Type u_3
      inst✝ : Quiver W'
      φ : Prefunctor V W'
      τ : W → W'
      h : ∀ (x : V), Eq (φ.obj x) (τ (σ x))
      X Y : V
      f : Quiver.Hom X Y
      ⊢ HEq (Eq.rec (Eq.rec (φ.map f) ⋯) ⋯) (φ.map f)
    -/
    apply (eqRec_heq _ _).trans
    have : ∀ {α γ} {β : α → γ → Sort _} {a a'} (p : a = a') g (b : β a g), HEq (p ▸ b) b := by
      intros
      subst_vars
      rfl
    /-
      case h_map.h.h
      V : Type u_1
      inst✝¹ : Quiver V
      W : Type u_2
      σ : V → W
      W' : Type u_3
      inst✝ : Quiver W'
      φ : Prefunctor V W'
      τ : W → W'
      h : ∀ (x : V), Eq (φ.obj x) (τ (σ x))
      X Y : V
      f : Quiver.Hom X Y
      this : ∀ {α : Sort ?u.2456} {γ : Sort ?u.2458} {β : α → γ → Sort ?u.2460} {a a …
      ⊢ HEq (Eq.rec (φ.map f) ⋯) (φ.map f)
    -/
    apply this
    /-
      🎉 no goals
    -/


theorem lift_unique (Φ : Push σ ⥤q W') (Φ₀ : Φ.obj = τ) (Φcomp : (of σ ⋙q Φ) = φ) :
    Φ = lift σ φ τ h := by
  /-
    V : Type u_1
    inst✝¹ : Quiver V
    W : Type u_2
    σ : V → W
    W' : Type u_3
    inst✝ : Quiver W'
    φ : Prefunctor V W'
    τ : W → W'
    h : ∀ (x : V), Eq (φ.obj x) (τ (σ x))
    Φ : Prefunctor (Quiver.Push σ) W'
    Φ₀ : Eq Φ.obj τ
    Φcomp : Eq ((Quiver.Push.of σ).comp Φ) φ
    ⊢ Eq Φ (Quiver.Push.lift σ φ τ h)
  -/
  dsimp only [of, lift]
  /-
    V : Type u_1
    inst✝¹ : Quiver V
    W : Type u_2
    σ : V → W
    W' : Type u_3
    inst✝ : Quiver W'
    φ : Prefunctor V W'
    τ : W → W'
    h : ∀ (x : V), Eq (φ.obj x) (τ (σ x))
    Φ : Prefunctor (Quiver.Push σ) W'
    Φ₀ : Eq Φ.obj τ
    Φcomp : Eq ((Quiver.Push.of σ).comp Φ) φ
    ⊢ Eq Φ { obj := τ, map := @Quiver.PushQuiver.rec V inst✝¹ W σ (fun X Y x => Qu …
  -/
  fapply Prefunctor.ext
    /-
      case h_obj
      V : Type u_1
      inst✝¹ : Quiver V
      W : Type u_2
      σ : V → W
      W' : Type u_3
      inst✝ : Quiver W'
      φ : Prefunctor V W'
      τ : W → W'
      h : ∀ (x : V), Eq (φ.obj x) (τ (σ x))
      Φ : Prefunctor (Quiver.Push σ) W'
      Φ₀ : Eq Φ.obj τ
      Φcomp : Eq ((Quiver.Push.of σ).comp Φ) φ
      ⊢ ∀ (X : Quiver.Push σ), Eq (Φ.obj X) ({ obj := τ, map := @Quiver.PushQuiver.r …
    -/
  · intro X
    /-
      case h_obj
      V : Type u_1
      inst✝¹ : Quiver V
      W : Type u_2
      σ : V → W
      W' : Type u_3
      inst✝ : Quiver W'
      φ : Prefunctor V W'
      τ : W → W'
      h : ∀ (x : V), Eq (φ.obj x) (τ (σ x))
      Φ : Prefunctor (Quiver.Push σ) W'
      Φ₀ : Eq Φ.obj τ
      Φcomp : Eq ((Quiver.Push.of σ).comp Φ) φ
      X : Quiver.Push σ
      ⊢ Eq (Φ.obj X) ({ obj := τ, map := @Quiver.PushQuiver.rec V inst✝¹ W σ (fun X  …
    -/
    simp only
    /-
      case h_obj
      V : Type u_1
      inst✝¹ : Quiver V
      W : Type u_2
      σ : V → W
      W' : Type u_3
      inst✝ : Quiver W'
      φ : Prefunctor V W'
      τ : W → W'
      h : ∀ (x : V), Eq (φ.obj x) (τ (σ x))
      Φ : Prefunctor (Quiver.Push σ) W'
      Φ₀ : Eq Φ.obj τ
      Φcomp : Eq ((Quiver.Push.of σ).comp Φ) φ
      X : Quiver.Push σ
      ⊢ Eq (Φ.obj X) (τ X)
    -/
    rw [Φ₀]
    /-
      🎉 no goals
    -/
    /-
      case h_map
      V : Type u_1
      inst✝¹ : Quiver V
      W : Type u_2
      σ : V → W
      W' : Type u_3
      inst✝ : Quiver W'
      φ : Prefunctor V W'
      τ : W → W'
      h : ∀ (x : V), Eq (φ.obj x) (τ (σ x))
      Φ : Prefunctor (Quiver.Push σ) W'
      Φ₀ : Eq Φ.obj τ
      Φcomp : Eq ((Quiver.Push.of σ).comp Φ) φ
      ⊢ ∀ (X Y : Quiver.Push σ) (f : Quiver.Hom X Y), Eq (Φ.map f) (Eq.recOn ⋯ (Eq.r …
    -/
  · rintro _ _ ⟨⟩
    /-
      case h_map.arrow
      V : Type u_1
      inst✝¹ : Quiver V
      W : Type u_2
      σ : V → W
      W' : Type u_3
      inst✝ : Quiver W'
      φ : Prefunctor V W'
      τ : W → W'
      h : ∀ (x : V), Eq (φ.obj x) (τ (σ x))
      Φ : Prefunctor (Quiver.Push σ) W'
      Φ₀ : Eq Φ.obj τ
      Φcomp : Eq ((Quiver.Push.of σ).comp Φ) φ
      X✝ Y✝ : V
      f✝ : Quiver.Hom X✝ Y✝
      ⊢ Eq (Φ.map (Quiver.PushQuiver.arrow f✝)) (Eq.recOn ⋯ (Eq.recOn ⋯ ({ obj := τ, …
    -/
    subst_vars
    /-
      case h_map.arrow
      V : Type u_1
      inst✝¹ : Quiver V
      W : Type u_2
      σ : V → W
      W' : Type u_3
      inst✝ : Quiver W'
      Φ : Prefunctor (Quiver.Push σ) W'
      X✝ Y✝ : V
      f✝ : Quiver.Hom X✝ Y✝
      h : ∀ (x : V), Eq (((Quiver.Push.of σ).comp Φ).obj x) (Φ.obj (σ x))
      ⊢ Eq (Φ.map (Quiver.PushQuiver.arrow f✝)) (Eq.recOn ⋯ (Eq.recOn ⋯ ({ obj := Φ. …
    -/
    simp only [Prefunctor.comp_map, cast_eq]
    /-
      case h_map.arrow
      V : Type u_1
      inst✝¹ : Quiver V
      W : Type u_2
      σ : V → W
      W' : Type u_3
      inst✝ : Quiver W'
      Φ : Prefunctor (Quiver.Push σ) W'
      X✝ Y✝ : V
      f✝ : Quiver.Hom X✝ Y✝
      h : ∀ (x : V), Eq (((Quiver.Push.of σ).comp Φ).obj x) (Φ.obj (σ x))
      ⊢ Eq (Φ.map (Quiver.PushQuiver.arrow f✝)) (id (⋯.mpr (⋯.mpr (Φ.map ((Quiver.Pu …
    -/
    rfl
    /-
      🎉 no goals
    -/


