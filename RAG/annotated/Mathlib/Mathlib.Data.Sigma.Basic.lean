instance instInhabitedSigma [Inhabited α] [Inhabited (β default)] : Inhabited (Sigma β) :=
  ⟨⟨default, default⟩⟩


instance instDecidableEqSigma [h₁ : DecidableEq α] [h₂ : ∀ a, DecidableEq (β a)] :
    DecidableEq (Sigma β)
  | ⟨a₁, b₁⟩, ⟨a₂, b₂⟩ =>
    match a₁, b₁, a₂, b₂, h₁ a₁ a₂ with
    | _, b₁, _, b₂, isTrue (Eq.refl _) =>
      match b₁, b₂, h₂ _ b₁ b₂ with
      | _, _, isTrue (Eq.refl _) => isTrue rfl
      | _, _, isFalse n => isFalse fun h ↦ Sigma.noConfusion h fun _ e₂ ↦ n <| eq_of_heq e₂
    | _, _, _, _, isFalse n => isFalse fun h ↦ Sigma.noConfusion h fun e₁ _ ↦ n e₁

-- sometimes the built-in injectivity support does not work

@[simp] -- @[nolint simpNF]
theorem mk.inj_iff {a₁ a₂ : α} {b₁ : β a₁} {b₂ : β a₂} :
    Sigma.mk a₁ b₁ = ⟨a₂, b₂⟩ ↔ a₁ = a₂ ∧ HEq b₁ b₂ :=
              /-
                α : Type u_1
                β : α → Type u_4
                a₁ a₂ : α
                b₁ : β a₁
                b₂ : β a₂
                h : Eq ⟨a₁, b₁⟩ ⟨a₂, b₂⟩
                ⊢ And (Eq a₁ a₂) (HEq b₁ b₂)
              -/
  ⟨fun h ↦ by cases h; simp,
                       /-
                         🎉 no goals
                       -/
                     /-
                       α : Type u_1
                       β : α → Type u_4
                       a₁ a₂ : α
                       b₁ : β a₁
                       b₂ : β a₂
                       x✝ : And (Eq a₁ a₂) (HEq b₁ b₂)
                       h₁ : Eq a₁ a₂
                       h₂ : HEq b₁ b₂
                       ⊢ Eq ⟨a₁, b₁⟩ ⟨a₂, b₂⟩
                     -/
   fun ⟨h₁, h₂⟩ ↦ by subst h₁; rw [eq_of_heq h₂]⟩
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem eta : ∀ x : Σa, β a, Sigma.mk x.1 x.2 = x
  | ⟨_, _⟩ => rfl


protected theorem eq {α : Type*} {β : α → Type*} : ∀ {p₁ p₂ : Σ a, β a} (h₁ : p₁.1 = p₂.1),
    (Eq.recOn h₁ p₁.2 : β p₂.1) = p₂.2 → p₁ = p₂
  | ⟨_, _⟩, _, rfl, rfl => rfl


/-- A version of `Iff.mp Sigma.ext_iff` for functions from a nonempty type to a sigma type. -/
theorem _root_.Function.eq_of_sigmaMk_comp {γ : Type*} [Nonempty γ]
    {a b : α} {f : γ → β a} {g : γ → β b} (h : Sigma.mk a ∘ f = Sigma.mk b ∘ g) :
    a = b ∧ HEq f g := by
  /-
    α : Type u_1
    β : α → Type u_4
    γ : Type u_7
    inst✝ : Nonempty γ
    a b : α
    f : γ → β a
    g : γ → β b
    h : Eq (Function.comp (Sigma.mk a) f) (Function.comp (Sigma.mk b) g)
    ⊢ And (Eq a b) (HEq f g)
  -/
  rcases ‹Nonempty γ› with ⟨i⟩
  /-
    case intro
    α : Type u_1
    β : α → Type u_4
    γ : Type u_7
    inst✝ : Nonempty γ
    a b : α
    f : γ → β a
    g : γ → β b
    h : Eq (Function.comp (Sigma.mk a) f) (Function.comp (Sigma.mk b) g)
    i : γ
    ⊢ And (Eq a b) (HEq f g)
  -/
  obtain rfl : a = b := congr_arg Sigma.fst (congr_fun h i)
  /-
    case intro
    α : Type u_1
    β : α → Type u_4
    γ : Type u_7
    inst✝ : Nonempty γ
    a : α
    f : γ → β a
    i : γ
    g : γ → β a
    h : Eq (Function.comp (Sigma.mk a) f) (Function.comp (Sigma.mk a) g)
    ⊢ And (Eq a a) (HEq f g)
  -/
  simpa [funext_iff] using h
  /-
    🎉 no goals
  -/


/-- A specialized ext lemma for equality of sigma types over an indexed subtype. -/
@[ext]
theorem subtype_ext {β : Type*} {p : α → β → Prop} :
    ∀ {x₀ x₁ : Σa, Subtype (p a)}, x₀.fst = x₁.fst → (x₀.snd : β) = x₁.snd → x₀ = x₁
  | ⟨_, _, _⟩, ⟨_, _, _⟩, rfl, rfl => rfl

-- This is not a good simp lemma, as its discrimination tree key is just an arrow.

theorem «forall» {p : (Σa, β a) → Prop} : (∀ x, p x) ↔ ∀ a b, p ⟨a, b⟩ :=
  ⟨fun h a b ↦ h ⟨a, b⟩, fun h ⟨a, b⟩ ↦ h a b⟩


@[simp]
theorem «exists» {p : (Σa, β a) → Prop} : (∃ x, p x) ↔ ∃ a b, p ⟨a, b⟩ :=
  ⟨fun ⟨⟨a, b⟩, h⟩ ↦ ⟨a, b, h⟩, fun ⟨a, b, h⟩ ↦ ⟨⟨a, b⟩, h⟩⟩


lemma exists' {p : ∀ a, β a → Prop} : (∃ a b, p a b) ↔ ∃ x : Σ a, β a, p x.1 x.2 :=
  (Sigma.exists (p := fun x ↦ p x.1 x.2)).symm


lemma forall' {p : ∀ a, β a → Prop} : (∀ a b, p a b) ↔ ∀ x : Σ a, β a, p x.1 x.2 :=
  (Sigma.forall (p := fun x ↦ p x.1 x.2)).symm


theorem _root_.sigma_mk_injective {i : α} : Injective (@Sigma.mk α β i)
  | _, _, rfl => rfl


theorem fst_surjective [h : ∀ a, Nonempty (β a)] : Surjective (fst : (Σ a, β a) → α) := fun a ↦
  let ⟨b⟩ := h a; ⟨⟨a, b⟩, rfl⟩


theorem fst_surjective_iff : Surjective (fst : (Σ a, β a) → α) ↔ ∀ a, Nonempty (β a) :=
  ⟨fun h a ↦ let ⟨x, hx⟩ := h a; hx ▸ ⟨x.2⟩, @fst_surjective _ _⟩


theorem fst_injective [h : ∀ a, Subsingleton (β a)] : Injective (fst : (Σ a, β a) → α) := by
  /-
    α : Type u_1
    β : α → Type u_4
    h : ∀ (a : α), Subsingleton (β a)
    ⊢ Function.Injective Sigma.fst
  -/
  rintro ⟨a₁, b₁⟩ ⟨a₂, b₂⟩ (rfl : a₁ = a₂)
  /-
    case mk.mk
    α : Type u_1
    β : α → Type u_4
    h : ∀ (a : α), Subsingleton (β a)
    a₁ : α
    b₁ b₂ : β a₁
    ⊢ Eq ⟨a₁, b₁⟩ ⟨a₁, b₂⟩
  -/
  exact congr_arg (mk a₁) <| Subsingleton.elim _ _
  /-
    🎉 no goals
  -/


theorem fst_injective_iff : Injective (fst : (Σ a, β a) → α) ↔ ∀ a, Subsingleton (β a) :=
  ⟨fun h _ ↦ ⟨fun _ _ ↦ sigma_mk_injective <| h rfl⟩, @fst_injective _ _⟩


/-- Map the left and right components of a sigma -/
def map (f₁ : α₁ → α₂) (f₂ : ∀ a, β₁ a → β₂ (f₁ a)) (x : Sigma β₁) : Sigma β₂ :=
  ⟨f₁ x.1, f₂ x.1 x.2⟩


lemma map_mk (f₁ : α₁ → α₂) (f₂ : ∀ a, β₁ a → β₂ (f₁ a)) (x : α₁) (y : β₁ x) :
    map f₁ f₂ ⟨x, y⟩ = ⟨f₁ x, f₂ x y⟩ := rfl

theorem Function.Injective.sigma_map {f₁ : α₁ → α₂} {f₂ : ∀ a, β₁ a → β₂ (f₁ a)}
    (h₁ : Injective f₁) (h₂ : ∀ a, Injective (f₂ a)) : Injective (Sigma.map f₁ f₂)
  | ⟨i, x⟩, ⟨j, y⟩, h => by
    /-
      α₁ : Type u_2
      α₂ : Type u_3
      β₁ : α₁ → Type u_5
      β₂ : α₂ → Type u_6
      f₁ : α₁ → α₂
      f₂ : (a : α₁) → β₁ a → β₂ (f₁ a)
      h₁ : Function.Injective f₁
      h₂ : ∀ (a : α₁), Function.Injective (f₂ a)
      i : α₁
      x : β₁ i
      j : α₁
      y : β₁ j
      h : Eq (Sigma.map f₁ f₂ ⟨i, x⟩) (Sigma.map f₁ f₂ ⟨j, y⟩)
      ⊢ Eq ⟨i, x⟩ ⟨j, y⟩
    -/
    obtain rfl : i = j := h₁ (Sigma.mk.inj_iff.mp h).1
    /-
      α₁ : Type u_2
      α₂ : Type u_3
      β₁ : α₁ → Type u_5
      β₂ : α₂ → Type u_6
      f₁ : α₁ → α₂
      f₂ : (a : α₁) → β₁ a → β₂ (f₁ a)
      h₁ : Function.Injective f₁
      h₂ : ∀ (a : α₁), Function.Injective (f₂ a)
      i : α₁
      x y : β₁ i
      h : Eq (Sigma.map f₁ f₂ ⟨i, x⟩) (Sigma.map f₁ f₂ ⟨i, y⟩)
      ⊢ Eq ⟨i, x⟩ ⟨i, y⟩
    -/
    obtain rfl : x = y := h₂ i (sigma_mk_injective h)
    /-
      α₁ : Type u_2
      α₂ : Type u_3
      β₁ : α₁ → Type u_5
      β₂ : α₂ → Type u_6
      f₁ : α₁ → α₂
      f₂ : (a : α₁) → β₁ a → β₂ (f₁ a)
      h₁ : Function.Injective f₁
      h₂ : ∀ (a : α₁), Function.Injective (f₂ a)
      i : α₁
      x : β₁ i
      h : Eq (Sigma.map f₁ f₂ ⟨i, x⟩) (Sigma.map f₁ f₂ ⟨i, x⟩)
      ⊢ Eq ⟨i, x⟩ ⟨i, x⟩
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem Function.Injective.of_sigma_map {f₁ : α₁ → α₂} {f₂ : ∀ a, β₁ a → β₂ (f₁ a)}
    (h : Injective (Sigma.map f₁ f₂)) (a : α₁) : Injective (f₂ a) := fun x y hxy ↦
  sigma_mk_injective <| @h ⟨a, x⟩ ⟨a, y⟩ (Sigma.ext rfl (heq_of_eq hxy))


theorem Function.Injective.sigma_map_iff {f₁ : α₁ → α₂} {f₂ : ∀ a, β₁ a → β₂ (f₁ a)}
    (h₁ : Injective f₁) : Injective (Sigma.map f₁ f₂) ↔ ∀ a, Injective (f₂ a) :=
  ⟨fun h ↦ h.of_sigma_map, h₁.sigma_map⟩


theorem Function.Surjective.sigma_map {f₁ : α₁ → α₂} {f₂ : ∀ a, β₁ a → β₂ (f₁ a)}
    (h₁ : Surjective f₁) (h₂ : ∀ a, Surjective (f₂ a)) : Surjective (Sigma.map f₁ f₂) := by
  /-
    α₁ : Type u_2
    α₂ : Type u_3
    β₁ : α₁ → Type u_5
    β₂ : α₂ → Type u_6
    f₁ : α₁ → α₂
    f₂ : (a : α₁) → β₁ a → β₂ (f₁ a)
    h₁ : Function.Surjective f₁
    h₂ : ∀ (a : α₁), Function.Surjective (f₂ a)
    ⊢ Function.Surjective (Sigma.map f₁ f₂)
  -/
  simp only [Surjective, Sigma.forall, h₁.forall]
  /-
    α₁ : Type u_2
    α₂ : Type u_3
    β₁ : α₁ → Type u_5
    β₂ : α₂ → Type u_6
    f₁ : α₁ → α₂
    f₂ : (a : α₁) → β₁ a → β₂ (f₁ a)
    h₁ : Function.Surjective f₁
    h₂ : ∀ (a : α₁), Function.Surjective (f₂ a)
    ⊢ ∀ (x : α₁) (b : β₂ (f₁ x)), Exists fun a => Eq (Sigma.map f₁ f₂ a) ⟨f₁ x, b⟩
  -/
  exact fun i ↦ (h₂ _).forall.2 fun x ↦ ⟨⟨i, x⟩, rfl⟩
  /-
    🎉 no goals
  -/


/-- Interpret a function on `Σ x : α, β x` as a dependent function with two arguments.

This also exists as an `Equiv` as `Equiv.piCurry γ`. -/
def Sigma.curry {γ : ∀ a, β a → Type*} (f : ∀ x : Sigma β, γ x.1 x.2) (x : α) (y : β x) : γ x y :=
  f ⟨x, y⟩


/-- Interpret a dependent function with two arguments as a function on `Σ x : α, β x`.

This also exists as an `Equiv` as `(Equiv.piCurry γ).symm`. -/
def Sigma.uncurry {γ : ∀ a, β a → Type*} (f : ∀ (x) (y : β x), γ x y) (x : Sigma β) : γ x.1 x.2 :=
  f x.1 x.2


@[simp]
theorem Sigma.uncurry_curry {γ : ∀ a, β a → Type*} (f : ∀ x : Sigma β, γ x.1 x.2) :
    Sigma.uncurry (Sigma.curry f) = f :=
  funext fun ⟨_, _⟩ ↦ rfl


@[simp]
theorem Sigma.curry_uncurry {γ : ∀ a, β a → Type*} (f : ∀ (x) (y : β x), γ x y) :
    Sigma.curry (Sigma.uncurry f) = f :=
  rfl


theorem Sigma.curry_update {γ : ∀ a, β a → Type*} [DecidableEq α] [∀ a, DecidableEq (β a)]
    (i : Σ a, β a) (f : (i : Σ a, β a) → γ i.1 i.2) (x : γ i.1 i.2) :
    Sigma.curry (Function.update f i x) =
      Function.update (Sigma.curry f) i.1 (Function.update (Sigma.curry f i.1) i.2 x) := by
  /-
    α : Type u_1
    β : α → Type u_4
    γ : (a : α) → β a → Type u_7
    inst✝¹ : DecidableEq α
    inst✝ : (a : α) → DecidableEq (β a)
    i : Sigma fun a => β a
    f : (i : Sigma fun a => β a) → γ i.fst i.snd
    x : γ i.fst i.snd
    ⊢ Eq (Sigma.curry (Function.update f i x)) (Function.update (Sigma.curry f) i. …
  -/
  obtain ⟨ia, ib⟩ := i
  /-
    case mk
    α : Type u_1
    β : α → Type u_4
    γ : (a : α) → β a → Type u_7
    inst✝¹ : DecidableEq α
    inst✝ : (a : α) → DecidableEq (β a)
    f : (i : Sigma fun a => β a) → γ i.fst i.snd
    ia : α
    ib : β ia
    x : γ ⟨ia, ib⟩.fst ⟨ia, ib⟩.snd
    ⊢ Eq (Sigma.curry (Function.update f ⟨ia, ib⟩ x)) (Function.update (Sigma.curr …
  -/
  ext ja jb
  /-
    case mk.h.h
    α : Type u_1
    β : α → Type u_4
    γ : (a : α) → β a → Type u_7
    inst✝¹ : DecidableEq α
    inst✝ : (a : α) → DecidableEq (β a)
    f : (i : Sigma fun a => β a) → γ i.fst i.snd
    ia : α
    ib : β ia
    x : γ ⟨ia, ib⟩.fst ⟨ia, ib⟩.snd
    ja : α
    jb : β ja
    ⊢ Eq (Sigma.curry (Function.update f ⟨ia, ib⟩ x) ja jb) (Function.update (Sigm …
  -/
  unfold Sigma.curry
  /-
    case mk.h.h
    α : Type u_1
    β : α → Type u_4
    γ : (a : α) → β a → Type u_7
    inst✝¹ : DecidableEq α
    inst✝ : (a : α) → DecidableEq (β a)
    f : (i : Sigma fun a => β a) → γ i.fst i.snd
    ia : α
    ib : β ia
    x : γ ⟨ia, ib⟩.fst ⟨ia, ib⟩.snd
    ja : α
    jb : β ja
    ⊢ Eq (Function.update f ⟨ia, ib⟩ x ⟨ja, jb⟩) (Function.update (fun x y => f ⟨x …
  -/
  obtain rfl | ha := eq_or_ne ia ja
    /-
      case mk.h.h.inl
      α : Type u_1
      β : α → Type u_4
      γ : (a : α) → β a → Type u_7
      inst✝¹ : DecidableEq α
      inst✝ : (a : α) → DecidableEq (β a)
      f : (i : Sigma fun a => β a) → γ i.fst i.snd
      ia : α
      ib : β ia
      x : γ ⟨ia, ib⟩.fst ⟨ia, ib⟩.snd
      jb : β ia
      ⊢ Eq (Function.update f ⟨ia, ib⟩ x ⟨ia, jb⟩) (Function.update (fun x y => f ⟨x …
    -/
  · obtain rfl | hb := eq_or_ne ib jb
      /-
        case mk.h.h.inl.inl
        α : Type u_1
        β : α → Type u_4
        γ : (a : α) → β a → Type u_7
        inst✝¹ : DecidableEq α
        inst✝ : (a : α) → DecidableEq (β a)
        f : (i : Sigma fun a => β a) → γ i.fst i.snd
        ia : α
        ib : β ia
        x : γ ⟨ia, ib⟩.fst ⟨ia, ib⟩.snd
        ⊢ Eq (Function.update f ⟨ia, ib⟩ x ⟨ia, ib⟩) (Function.update (fun x y => f ⟨x …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case mk.h.h.inl.inr
        α : Type u_1
        β : α → Type u_4
        γ : (a : α) → β a → Type u_7
        inst✝¹ : DecidableEq α
        inst✝ : (a : α) → DecidableEq (β a)
        f : (i : Sigma fun a => β a) → γ i.fst i.snd
        ia : α
        ib : β ia
        x : γ ⟨ia, ib⟩.fst ⟨ia, ib⟩.snd
        jb : β ia
        hb : Ne ib jb
        ⊢ Eq (Function.update f ⟨ia, ib⟩ x ⟨ia, jb⟩) (Function.update (fun x y => f ⟨x …
      -/
    · simp only [update_self]
      /-
        case mk.h.h.inl.inr
        α : Type u_1
        β : α → Type u_4
        γ : (a : α) → β a → Type u_7
        inst✝¹ : DecidableEq α
        inst✝ : (a : α) → DecidableEq (β a)
        f : (i : Sigma fun a => β a) → γ i.fst i.snd
        ia : α
        ib : β ia
        x : γ ⟨ia, ib⟩.fst ⟨ia, ib⟩.snd
        jb : β ia
        hb : Ne ib jb
        ⊢ Eq (Function.update f ⟨ia, ib⟩ x ⟨ia, jb⟩) (Function.update (fun y => f ⟨ia, …
      -/
      rw [Function.update_of_ne (mt _ hb.symm), Function.update_of_ne hb.symm]
      /-
        α : Type u_1
        β : α → Type u_4
        γ : (a : α) → β a → Type u_7
        inst✝¹ : DecidableEq α
        inst✝ : (a : α) → DecidableEq (β a)
        f : (i : Sigma fun a => β a) → γ i.fst i.snd
        ia : α
        ib : β ia
        x : γ ⟨ia, ib⟩.fst ⟨ia, ib⟩.snd
        jb : β ia
        hb : Ne ib jb
        ⊢ Eq ⟨ia, jb⟩ ⟨ia, ib⟩ → Eq jb ib
      -/
      rintro h
      /-
        α : Type u_1
        β : α → Type u_4
        γ : (a : α) → β a → Type u_7
        inst✝¹ : DecidableEq α
        inst✝ : (a : α) → DecidableEq (β a)
        f : (i : Sigma fun a => β a) → γ i.fst i.snd
        ia : α
        ib : β ia
        x : γ ⟨ia, ib⟩.fst ⟨ia, ib⟩.snd
        jb : β ia
        hb : Ne ib jb
        h : Eq ⟨ia, jb⟩ ⟨ia, ib⟩
        ⊢ Eq jb ib
      -/
      injection h
      /-
        🎉 no goals
      -/
    /-
      case mk.h.h.inr
      α : Type u_1
      β : α → Type u_4
      γ : (a : α) → β a → Type u_7
      inst✝¹ : DecidableEq α
      inst✝ : (a : α) → DecidableEq (β a)
      f : (i : Sigma fun a => β a) → γ i.fst i.snd
      ia : α
      ib : β ia
      x : γ ⟨ia, ib⟩.fst ⟨ia, ib⟩.snd
      ja : α
      jb : β ja
      ha : Ne ia ja
      ⊢ Eq (Function.update f ⟨ia, ib⟩ x ⟨ja, jb⟩) (Function.update (fun x y => f ⟨x …
    -/
  · rw [Function.update_of_ne (ne_of_apply_ne Sigma.fst _), Function.update_of_ne]
      /-
        case mk.h.h.inr.h
        α : Type u_1
        β : α → Type u_4
        γ : (a : α) → β a → Type u_7
        inst✝¹ : DecidableEq α
        inst✝ : (a : α) → DecidableEq (β a)
        f : (i : Sigma fun a => β a) → γ i.fst i.snd
        ia : α
        ib : β ia
        x : γ ⟨ia, ib⟩.fst ⟨ia, ib⟩.snd
        ja : α
        jb : β ja
        ha : Ne ia ja
        ⊢ Ne ja ⟨ia, ib⟩.fst
      -/
    · exact ha.symm
      /-
        🎉 no goals
      -/
      /-
        α : Type u_1
        β : α → Type u_4
        γ : (a : α) → β a → Type u_7
        inst✝¹ : DecidableEq α
        inst✝ : (a : α) → DecidableEq (β a)
        f : (i : Sigma fun a => β a) → γ i.fst i.snd
        ia : α
        ib : β ia
        x : γ ⟨ia, ib⟩.fst ⟨ia, ib⟩.snd
        ja : α
        jb : β ja
        ha : Ne ia ja
        ⊢ Ne ⟨ja, jb⟩.fst ⟨ia, ib⟩.fst
      -/
    · exact ha.symm
      /-
        🎉 no goals
      -/


/-- Convert a product type to a Σ-type. -/
def Prod.toSigma {α β} (p : α × β) : Σ_ : α, β :=
  ⟨p.1, p.2⟩


@[simp]
theorem Prod.fst_comp_toSigma {α β} : Sigma.fst ∘ @Prod.toSigma α β = Prod.fst :=
  rfl


@[simp]
theorem Prod.fst_toSigma {α β} (x : α × β) : (Prod.toSigma x).fst = x.fst :=
  rfl


@[simp]
theorem Prod.snd_toSigma {α β} (x : α × β) : (Prod.toSigma x).snd = x.snd :=
  rfl


@[simp]
theorem Prod.toSigma_mk {α β} (x : α) (y : β) : (x, y).toSigma = ⟨x, y⟩ :=
  rfl

-- Porting note: the meta instance `has_reflect (Σa, β a)` was removed here.


/-- Nondependent eliminator for `PSigma`. -/
def elim {γ} (f : ∀ a, β a → γ) (a : PSigma β) : γ :=
  PSigma.casesOn a f


@[simp]
theorem elim_val {γ} (f : ∀ a, β a → γ) (a b) : PSigma.elim f ⟨a, b⟩ = f a b :=
  rfl


@[deprecated (since := "2024-07-27")] alias ex_of_psig := ex_of_PSigma


instance [Inhabited α] [Inhabited (β default)] : Inhabited (PSigma β) :=
  ⟨⟨default, default⟩⟩


instance decidableEq [h₁ : DecidableEq α] [h₂ : ∀ a, DecidableEq (β a)] : DecidableEq (PSigma β)
  | ⟨a₁, b₁⟩, ⟨a₂, b₂⟩ =>
    match a₁, b₁, a₂, b₂, h₁ a₁ a₂ with
    | _, b₁, _, b₂, isTrue (Eq.refl _) =>
      match b₁, b₂, h₂ _ b₁ b₂ with
      | _, _, isTrue (Eq.refl _) => isTrue rfl
      | _, _, isFalse n => isFalse fun h ↦ PSigma.noConfusion h fun _ e₂ ↦ n <| eq_of_heq e₂
    | _, _, _, _, isFalse n => isFalse fun h ↦ PSigma.noConfusion h fun e₁ _ ↦ n e₁

-- See https://leanprover.zulipchat.com/#narrow/stream/287929-mathlib4/topic/porting.20data.2Esigma.2Ebasic/near/304855864
-- for an explanation of why this is currently needed. It generates `PSigma.mk.inj`.
-- This could be done elsewhere.

gen_injective_theorems% PSigma


theorem mk.inj_iff {a₁ a₂ : α} {b₁ : β a₁} {b₂ : β a₂} :
    @PSigma.mk α β a₁ b₁ = @PSigma.mk α β a₂ b₂ ↔ a₁ = a₂ ∧ HEq b₁ b₂ :=
  (Iff.intro PSigma.mk.inj) fun ⟨h₁, h₂⟩ ↦
    match a₁, a₂, b₁, b₂, h₁, h₂ with
    | _, _, _, _, Eq.refl _, HEq.refl _ => rfl


@[deprecated PSigma.ext_iff (since := "2024-07-27")]
protected theorem eq {α : Sort*} {β : α → Sort*} : ∀ {p₁ p₂ : Σ' a, β a} (h₁ : p₁.1 = p₂.1),
    (Eq.recOn h₁ p₁.2 : β p₂.1) = p₂.2 → p₁ = p₂
  | ⟨_, _⟩, _, rfl, rfl => rfl

-- This should not be a simp lemma, since its discrimination tree key would just be `→`.

theorem «forall» {p : (Σ'a, β a) → Prop} : (∀ x, p x) ↔ ∀ a b, p ⟨a, b⟩ :=
  ⟨fun h a b ↦ h ⟨a, b⟩, fun h ⟨a, b⟩ ↦ h a b⟩


@[simp] lemma «exists» {p : (Σ' a, β a) → Prop} : (∃ x, p x) ↔ ∃ a b, p ⟨a, b⟩ :=
  ⟨fun ⟨⟨a, b⟩, h⟩ ↦ ⟨a, b, h⟩, fun ⟨a, b, h⟩ ↦ ⟨⟨a, b⟩, h⟩⟩


/-- A specialized ext lemma for equality of `PSigma` types over an indexed subtype. -/
@[ext]
theorem subtype_ext {β : Sort*} {p : α → β → Prop} :
    ∀ {x₀ x₁ : Σ'a, Subtype (p a)}, x₀.fst = x₁.fst → (x₀.snd : β) = x₁.snd → x₀ = x₁
  | ⟨_, _, _⟩, ⟨_, _, _⟩, rfl, rfl => rfl


/-- Map the left and right components of a sigma -/
def map (f₁ : α₁ → α₂) (f₂ : ∀ a, β₁ a → β₂ (f₁ a)) : PSigma β₁ → PSigma β₂
  | ⟨a, b⟩ => ⟨f₁ a, f₂ a b⟩


