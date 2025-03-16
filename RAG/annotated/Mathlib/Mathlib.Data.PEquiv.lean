/-- A `PEquiv` is a partial equivalence, a representation of a bijection between a subset
  of `α` and a subset of `β`. See also `PartialEquiv` for a version that requires `toFun` and
`invFun` to be globally defined functions and has `source` and `target` sets as extra fields. -/
structure PEquiv (α : Type u) (β : Type v) where
  /-- The underlying partial function of a `PEquiv` -/
  toFun : α → Option β
  /-- The partial inverse of `toFun` -/
  invFun : β → Option α
  /-- `invFun` is the partial inverse of `toFun`  -/
  inv : ∀ (a : α) (b : β), a ∈ invFun b ↔ b ∈ toFun a


/-- A `PEquiv` is a partial equivalence, a representation of a bijection between a subset
  of `α` and a subset of `β`. See also `PartialEquiv` for a version that requires `toFun` and
`invFun` to be globally defined functions and has `source` and `target` sets as extra fields. -/
infixr:25 " ≃. " => PEquiv


instance : FunLike (α ≃. β) α (Option β) :=
  { coe := toFun
    coe_injective' := by
      /-
        α : Type u
        β : Type v
        γ : Type w
        δ : Type x
        ⊢ Function.Injective PEquiv.toFun
      -/
      rintro ⟨f₁, f₂, hf⟩ ⟨g₁, g₂, hg⟩ (rfl : f₁ = g₁)
      /-
        case mk.mk
        α : Type u
        β : Type v
        γ : Type w
        δ : Type x
        f₁ : α → Option β
        f₂ : β → Option α
        hf : ∀ (a : α) (b : β), Iff (Membership.mem (f₂ b) a) (Membership.mem (f₁ a) b)
        g₂ : β → Option α
        hg : ∀ (a : α) (b : β), Iff (Membership.mem (g₂ b) a) (Membership.mem (f₁ a) b)
        ⊢ Eq { toFun := f₁, invFun := f₂, inv := hf } { toFun := f₁, invFun := g₂, inv …
      -/
      congr with y x
      /-
        case mk.mk.e_invFun.h.a
        α : Type u
        β : Type v
        γ : Type w
        δ : Type x
        f₁ : α → Option β
        f₂ : β → Option α
        hf : ∀ (a : α) (b : β), Iff (Membership.mem (f₂ b) a) (Membership.mem (f₁ a) b)
        g₂ : β → Option α
        hg : ∀ (a : α) (b : β), Iff (Membership.mem (g₂ b) a) (Membership.mem (f₁ a) b)
        y : β
        x : α
        ⊢ Iff (Membership.mem (f₂ y) x) (Membership.mem (g₂ y) x)
      -/
      simp only [hf, hg] }
      /-
        🎉 no goals
      -/


@[simp] theorem coe_mk (f₁ : α → Option β) (f₂ h) : (mk f₁ f₂ h : α → Option β) = f₁ :=
  rfl


theorem coe_mk_apply (f₁ : α → Option β) (f₂ : β → Option α) (h) (x : α) :
    (PEquiv.mk f₁ f₂ h : α → Option β) x = f₁ x :=
  rfl


@[ext] theorem ext {f g : α ≃. β} (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext f g h


/-- The identity map as a partial equivalence. -/
@[refl]
protected def refl (α : Type*) : α ≃. α where
  toFun := some
  invFun := some
  inv _ _ := eq_comm


/-- The inverse partial equivalence. -/
@[symm]
protected def symm (f : α ≃. β) : β ≃. α where
  toFun := f.2
  invFun := f.1
  inv _ _ := (f.inv _ _).symm


theorem mem_iff_mem (f : α ≃. β) : ∀ {a : α} {b : β}, a ∈ f.symm b ↔ b ∈ f a :=
  f.3 _ _


theorem eq_some_iff (f : α ≃. β) : ∀ {a : α} {b : β}, f.symm b = some a ↔ f a = some b :=
  f.3 _ _


/-- Composition of partial equivalences `f : α ≃. β` and `g : β ≃. γ`. -/
@[trans]
protected def trans (f : α ≃. β) (g : β ≃. γ) :
    α ≃. γ where
  toFun a := (f a).bind g
  invFun a := (g.symm a).bind f.symm
                /-
                  α : Type u
                  β : Type v
                  γ : Type w
                  δ : Type x
                  f : PEquiv α β
                  g : PEquiv β γ
                  a : α
                  b : γ
                  ⊢ Iff (Membership.mem ((fun a => (g.symm a).bind ⇑f.symm) b) a) (Membership.me …
                -/
  inv a b := by simp_all [and_comm, eq_some_iff f, eq_some_iff g, bind_eq_some]
                /-
                  🎉 no goals
                -/


@[simp]
theorem refl_apply (a : α) : PEquiv.refl α a = some a :=
  rfl


@[simp]
theorem symm_refl : (PEquiv.refl α).symm = PEquiv.refl α :=
  rfl


@[simp]
theorem symm_symm (f : α ≃. β) : f.symm.symm = f := rfl


theorem symm_bijective : Function.Bijective (PEquiv.symm : (α ≃. β) → β ≃. α) :=
  Function.bijective_iff_has_inverse.mpr ⟨_, symm_symm, symm_symm⟩


theorem symm_injective : Function.Injective (@PEquiv.symm α β) :=
  symm_bijective.injective


theorem trans_assoc (f : α ≃. β) (g : β ≃. γ) (h : γ ≃. δ) :
    (f.trans g).trans h = f.trans (g.trans h) :=
  ext fun _ => Option.bind_assoc _ _ _


theorem mem_trans (f : α ≃. β) (g : β ≃. γ) (a : α) (c : γ) :
    c ∈ f.trans g a ↔ ∃ b, b ∈ f a ∧ c ∈ g b :=
  Option.bind_eq_some'


theorem trans_eq_some (f : α ≃. β) (g : β ≃. γ) (a : α) (c : γ) :
    f.trans g a = some c ↔ ∃ b, f a = some b ∧ g b = some c :=
  Option.bind_eq_some'


theorem trans_eq_none (f : α ≃. β) (g : β ≃. γ) (a : α) :
    f.trans g a = none ↔ ∀ b c, b ∉ f a ∨ c ∉ g b := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    f : PEquiv α β
    g : PEquiv β γ
    a : α
    ⊢ Iff (Eq ((f.trans g) a) Option.none) (∀ (b : β) (c : γ), Or (Not (Membership …
  -/
  simp only [eq_none_iff_forall_not_mem, mem_trans, imp_iff_not_or.symm]
  /-
    α : Type u
    β : Type v
    γ : Type w
    f : PEquiv α β
    g : PEquiv β γ
    a : α
    ⊢ Iff (∀ (a_1 : γ), Not (Exists fun b => And (Membership.mem (f a) b) (Members …
  -/
  push_neg
  /-
    α : Type u
    β : Type v
    γ : Type w
    f : PEquiv α β
    g : PEquiv β γ
    a : α
    ⊢ Iff (∀ (a_1 : γ) (b : β), Membership.mem (f a) b → Not (Membership.mem (g b) …
  -/
  exact forall_swap
  /-
    🎉 no goals
  -/


@[simp]
theorem refl_trans (f : α ≃. β) : (PEquiv.refl α).trans f = f := by
  /-
    α : Type u
    β : Type v
    f : PEquiv α β
    ⊢ Eq ((PEquiv.refl α).trans f) f
  -/
  ext; dsimp [PEquiv.trans]; rfl
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem trans_refl (f : α ≃. β) : f.trans (PEquiv.refl β) = f := by
  /-
    α : Type u
    β : Type v
    f : PEquiv α β
    ⊢ Eq (f.trans (PEquiv.refl β)) f
  -/
  ext; dsimp [PEquiv.trans]; simp
                             /-
                               🎉 no goals
                             -/


protected theorem inj (f : α ≃. β) {a₁ a₂ : α} {b : β} (h₁ : b ∈ f a₁) (h₂ : b ∈ f a₂) :
                  /-
                    α : Type u
                    β : Type v
                    f : PEquiv α β
                    a₁ a₂ : α
                    b : β
                    h₁ : Membership.mem (f a₁) b
                    h₂ : Membership.mem (f a₂) b
                    ⊢ Eq a₁ a₂
                  -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
    a₁ = a₂ := by rw [← mem_iff_mem] at *; cases h : f.symm b <;> simp_all
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- If the domain of a `PEquiv` is `α` except a point, its forward direction is injective. -/
theorem injective_of_forall_ne_isSome (f : α ≃. β) (a₂ : α)
    (h : ∀ a₁ : α, a₁ ≠ a₂ → isSome (f a₁)) : Injective f :=
  HasLeftInverse.injective
    ⟨fun b => Option.recOn b a₂ fun b' => Option.recOn (f.symm b') a₂ id, fun x => by
      classical
        cases hfx : f x
        · have : x = a₂ := not_imp_comm.1 (h x) (hfx.symm ▸ by simp)
          simp [this]
        · dsimp only
          rw [(eq_some_iff f).2 hfx]
          rfl⟩


/-- If the domain of a `PEquiv` is all of `α`, its forward direction is injective. -/
theorem injective_of_forall_isSome {f : α ≃. β} (h : ∀ a : α, isSome (f a)) : Injective f :=
  (Classical.em (Nonempty α)).elim
    (fun hn => injective_of_forall_ne_isSome f (Classical.choice hn) fun a _ => h a) fun hn x =>
    (hn ⟨x⟩).elim


/-- Creates a `PEquiv` that is the identity on `s`, and `none` outside of it. -/
def ofSet (s : Set α) [DecidablePred (· ∈ s)] :
    α ≃. α where
  toFun a := if a ∈ s then some a else none
  invFun a := if a ∈ s then some a else none
  inv a b := by
    /-
      α : Type u
      β : Type v
      γ : Type w
      δ : Type x
      s✝ : Set α
      inst✝¹ : DecidablePred fun x => Membership.mem s✝ x
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      a b : α
      ⊢ Iff (Membership.mem ((fun a => ite (Membership.mem s a) (Option.some a) Opti …
    -/
    dsimp only
    /-
      α : Type u
      β : Type v
      γ : Type w
      δ : Type x
      s✝ : Set α
      inst✝¹ : DecidablePred fun x => Membership.mem s✝ x
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      a b : α
      ⊢ Iff (Membership.mem (ite (Membership.mem s b) (Option.some b) Option.none) a …
    -/
    split_ifs with hb ha ha
      /-
        case pos
        α : Type u
        β : Type v
        γ : Type w
        δ : Type x
        s✝ : Set α
        inst✝¹ : DecidablePred fun x => Membership.mem s✝ x
        s : Set α
        inst✝ : DecidablePred fun x => Membership.mem s x
        a b : α
        hb : Membership.mem s b
        ha : Membership.mem s a
        ⊢ Iff (Membership.mem (Option.some b) a) (Membership.mem (Option.some a) b)
      -/
    · simp [eq_comm]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        β : Type v
        γ : Type w
        δ : Type x
        s✝ : Set α
        inst✝¹ : DecidablePred fun x => Membership.mem s✝ x
        s : Set α
        inst✝ : DecidablePred fun x => Membership.mem s x
        a b : α
        hb : Membership.mem s b
        ha : Not (Membership.mem s a)
        ⊢ Iff (Membership.mem (Option.some b) a) (Membership.mem Option.none b)
      -/
    · simp [ne_of_mem_of_not_mem hb ha]
      /-
        🎉 no goals
      -/
      /-
        case pos
        α : Type u
        β : Type v
        γ : Type w
        δ : Type x
        s✝ : Set α
        inst✝¹ : DecidablePred fun x => Membership.mem s✝ x
        s : Set α
        inst✝ : DecidablePred fun x => Membership.mem s x
        a b : α
        hb : Not (Membership.mem s b)
        ha : Membership.mem s a
        ⊢ Iff (Membership.mem Option.none a) (Membership.mem (Option.some a) b)
      -/
    · simp [ne_of_mem_of_not_mem ha hb]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        β : Type v
        γ : Type w
        δ : Type x
        s✝ : Set α
        inst✝¹ : DecidablePred fun x => Membership.mem s✝ x
        s : Set α
        inst✝ : DecidablePred fun x => Membership.mem s x
        a b : α
        hb : Not (Membership.mem s b)
        ha : Not (Membership.mem s a)
        ⊢ Iff (Membership.mem Option.none a) (Membership.mem Option.none b)
      -/
    · simp
      /-
        🎉 no goals
      -/


theorem mem_ofSet_self_iff {s : Set α} [DecidablePred (· ∈ s)] {a : α} : a ∈ ofSet s a ↔ a ∈ s := by
  /-
    α : Type u
    s : Set α
    inst✝ : DecidablePred fun x => Membership.mem s x
    a : α
    ⊢ Iff (Membership.mem ((PEquiv.ofSet s) a) a) (Membership.mem s a)
  -/
                               /-
                                 🎉 no goals
                               -/
  dsimp [ofSet]; split_ifs <;> simp [*]
                               /-
                                 🎉 no goals
                               -/


theorem mem_ofSet_iff {s : Set α} [DecidablePred (· ∈ s)] {a b : α} :
    a ∈ ofSet s b ↔ a = b ∧ a ∈ s := by
  /-
    α : Type u
    s : Set α
    inst✝ : DecidablePred fun x => Membership.mem s x
    a b : α
    ⊢ Iff (Membership.mem ((PEquiv.ofSet s) b) a) (And (Eq a b) (Membership.mem s  …
  -/
  dsimp [ofSet]
  /-
    α : Type u
    s : Set α
    inst✝ : DecidablePred fun x => Membership.mem s x
    a b : α
    ⊢ Iff (Membership.mem (ite (Membership.mem s b) (Option.some b) Option.none) a …
  -/
  split_ifs with h
    /-
      case pos
      α : Type u
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      a b : α
      h : Membership.mem s b
      ⊢ Iff (Membership.mem (Option.some b) a) (And (Eq a b) (Membership.mem s a))
    -/
  · simp only [mem_def, eq_comm, some.injEq, iff_self_and]
    /-
      case pos
      α : Type u
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      a b : α
      h : Membership.mem s b
      ⊢ Eq a b → Membership.mem s a
    -/
    rintro rfl
    /-
      case pos
      α : Type u
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      a : α
      h : Membership.mem s a
      ⊢ Membership.mem s a
    -/
    exact h
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      a b : α
      h : Not (Membership.mem s b)
      ⊢ Iff (Membership.mem Option.none a) (And (Eq a b) (Membership.mem s a))
    -/
  · simp only [mem_def, false_iff, not_and, reduceCtorEq]
    /-
      case neg
      α : Type u
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      a b : α
      h : Not (Membership.mem s b)
      ⊢ Eq a b → Not (Membership.mem s a)
    -/
    rintro rfl
    /-
      case neg
      α : Type u
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      a : α
      h : Not (Membership.mem s a)
      ⊢ Not (Membership.mem s a)
    -/
    exact h
    /-
      🎉 no goals
    -/


@[simp]
theorem ofSet_eq_some_iff {s : Set α} {_ : DecidablePred (· ∈ s)} {a b : α} :
    ofSet s b = some a ↔ a = b ∧ a ∈ s :=
  mem_ofSet_iff


theorem ofSet_eq_some_self_iff {s : Set α} {_ : DecidablePred (· ∈ s)} {a : α} :
    ofSet s a = some a ↔ a ∈ s :=
  mem_ofSet_self_iff


@[simp]
theorem ofSet_symm : (ofSet s).symm = ofSet s :=
  rfl


@[simp]
theorem ofSet_univ : ofSet Set.univ = PEquiv.refl α :=
  rfl


@[simp]
theorem ofSet_eq_refl {s : Set α} [DecidablePred (· ∈ s)] :
    ofSet s = PEquiv.refl α ↔ s = Set.univ :=
  ⟨fun h => by
    /-
      α : Type u
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      h : Eq (PEquiv.ofSet s) (PEquiv.refl α)
      ⊢ Eq s Set.univ
    -/
    rw [Set.eq_univ_iff_forall]
    /-
      α : Type u
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      h : Eq (PEquiv.ofSet s) (PEquiv.refl α)
      ⊢ ∀ (x : α), Membership.mem s x
    -/
    intro
    /-
      α : Type u
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      h : Eq (PEquiv.ofSet s) (PEquiv.refl α)
      x✝ : α
      ⊢ Membership.mem s x✝
    -/
    rw [← mem_ofSet_self_iff, h]
    /-
      α : Type u
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      h : Eq (PEquiv.ofSet s) (PEquiv.refl α)
      x✝ : α
      ⊢ Membership.mem ((PEquiv.refl α) x✝) x✝
    -/
    /-
      🎉 no goals
    -/
    exact rfl, fun h => by simp only [← ofSet_univ, h]⟩
                           /-
                             🎉 no goals
                           -/


theorem symm_trans_rev (f : α ≃. β) (g : β ≃. γ) : (f.trans g).symm = g.symm.trans f.symm :=
  rfl


theorem self_trans_symm (f : α ≃. β) : f.trans f.symm = ofSet { a | (f a).isSome } := by
  /-
    α : Type u
    β : Type v
    f : PEquiv α β
    ⊢ Eq (f.trans f.symm) (PEquiv.ofSet (setOf fun a => Eq (f a).isSome Bool.true))
  -/
  ext
  /-
    case h.a
    α : Type u
    β : Type v
    f : PEquiv α β
    x✝ a✝ : α
    ⊢ Iff (Membership.mem ((f.trans f.symm) x✝) a✝) (Membership.mem ((PEquiv.ofSet …
  -/
  dsimp [PEquiv.trans]
  simp only [eq_some_iff f, Option.isSome_iff_exists, Option.mem_def, bind_eq_some',
    ofSet_eq_some_iff]
  /-
    case h.a
    α : Type u
    β : Type v
    f : PEquiv α β
    x✝ a✝ : α
    ⊢ Iff (Exists fun a => And (Eq (f x✝) (Option.some a)) (Eq (f a✝) (Option.some …
  -/
  constructor
    /-
      case h.a.mp
      α : Type u
      β : Type v
      f : PEquiv α β
      x✝ a✝ : α
      ⊢ (Exists fun a => And (Eq (f x✝) (Option.some a)) (Eq (f a✝) (Option.some a)) …
    -/
  · rintro ⟨b, hb₁, hb₂⟩
    /-
      case h.a.mp.intro.intro
      α : Type u
      β : Type v
      f : PEquiv α β
      x✝ a✝ : α
      b : β
      hb₁ : Eq (f x✝) (Option.some b)
      hb₂ : Eq (f a✝) (Option.some b)
      ⊢ And (Eq a✝ x✝) (Membership.mem (setOf fun a => Exists fun a_1 => Eq (f a) (O …
    -/
    exact ⟨PEquiv.inj _ hb₂ hb₁, b, hb₂⟩
    /-
      🎉 no goals
    -/
    /-
      case h.a.mpr
      α : Type u
      β : Type v
      f : PEquiv α β
      x✝ a✝ : α
      ⊢ And (Eq a✝ x✝) (Membership.mem (setOf fun a => Exists fun a_1 => Eq (f a) (O …
    -/
  · simp +contextual
    /-
      🎉 no goals
    -/


theorem symm_trans_self (f : α ≃. β) : f.symm.trans f = ofSet { b | (f.symm b).isSome } :=
                       /-
                         α : Type u
                         β : Type v
                         f : PEquiv α β
                         ⊢ Eq (f.symm.trans f).symm (PEquiv.ofSet (setOf fun b => Eq (f.symm b).isSome  …
                       -/
  symm_injective <| by simp [symm_trans_rev, self_trans_symm, -symm_symm]
                       /-
                         🎉 no goals
                       -/


theorem trans_symm_eq_iff_forall_isSome {f : α ≃. β} :
    f.trans f.symm = PEquiv.refl α ↔ ∀ a, isSome (f a) := by
  /-
    α : Type u
    β : Type v
    f : PEquiv α β
    ⊢ Iff (Eq (f.trans f.symm) (PEquiv.refl α)) (∀ (a : α), Eq (f a).isSome Bool.t …
  -/
  rw [self_trans_symm, ofSet_eq_refl, Set.eq_univ_iff_forall]; rfl
                                                               /-
                                                                 🎉 no goals
                                                               -/


instance instBotPEquiv : Bot (α ≃. β) :=
  ⟨{  toFun := fun _ => none
      invFun := fun _ => none
                /-
                  α : Type u
                  β : Type v
                  γ : Type w
                  δ : Type x
                  ⊢ ∀ (a : α) (b : β), Iff (Membership.mem ((fun x => Option.none) b) a) (Member …
                -/
      inv := by simp }⟩
                /-
                  🎉 no goals
                -/


instance : Inhabited (α ≃. β) :=
  ⟨⊥⟩


@[simp]
theorem bot_apply (a : α) : (⊥ : α ≃. β) a = none :=
  rfl


@[simp]
theorem symm_bot : (⊥ : α ≃. β).symm = ⊥ :=
  rfl


@[simp]
theorem trans_bot (f : α ≃. β) : f.trans (⊥ : β ≃. γ) = ⊥ := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    f : PEquiv α β
    ⊢ Eq (f.trans Bot.bot) Bot.bot
  -/
  ext; dsimp [PEquiv.trans]; simp
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem bot_trans (f : β ≃. γ) : (⊥ : α ≃. β).trans f = ⊥ := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    f : PEquiv β γ
    ⊢ Eq (Bot.bot.trans f) Bot.bot
  -/
  ext; dsimp [PEquiv.trans]; simp
                             /-
                               🎉 no goals
                             -/


theorem isSome_symm_get (f : α ≃. β) {a : α} (h : isSome (f a)) :
    isSome (f.symm (Option.get _ h)) :=
                             /-
                               α : Type u
                               β : Type v
                               f : PEquiv α β
                               a : α
                               h : Eq (f a).isSome Bool.true
                               ⊢ Eq (f.symm ((f a).get h)) (Option.some a)
                             -/
  isSome_iff_exists.2 ⟨a, by rw [f.eq_some_iff, some_get]⟩
                             /-
                               🎉 no goals
                             -/


/-- Create a `PEquiv` which sends `a` to `b` and `b` to `a`, but is otherwise `none`. -/
def single (a : α) (b : β) :
    α ≃. β where
  toFun x := if x = a then some b else none
  invFun x := if x = b then some a else none
  inv x y := by
    /-
      α : Type u
      β : Type v
      γ : Type w
      δ : Type x
      inst✝² : DecidableEq α
      inst✝¹ : DecidableEq β
      inst✝ : DecidableEq γ
      a : α
      b : β
      x : α
      y : β
      ⊢ Iff (Membership.mem ((fun x => ite (Eq x b) (Option.some a) Option.none) y)  …
    -/
    dsimp only
    /-
      α : Type u
      β : Type v
      γ : Type w
      δ : Type x
      inst✝² : DecidableEq α
      inst✝¹ : DecidableEq β
      inst✝ : DecidableEq γ
      a : α
      b : β
      x : α
      y : β
      ⊢ Iff (Membership.mem (ite (Eq y b) (Option.some a) Option.none) x) (Membershi …
    -/
    split_ifs with h1 h2
      /-
        case pos
        α : Type u
        β : Type v
        γ : Type w
        δ : Type x
        inst✝² : DecidableEq α
        inst✝¹ : DecidableEq β
        inst✝ : DecidableEq γ
        a : α
        b : β
        x : α
        y : β
        h1 : Eq y b
        h2 : Eq x a
        ⊢ Iff (Membership.mem (Option.some a) x) (Membership.mem (Option.some b) y)
      -/
    · simp [*]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        β : Type v
        γ : Type w
        δ : Type x
        inst✝² : DecidableEq α
        inst✝¹ : DecidableEq β
        inst✝ : DecidableEq γ
        a : α
        b : β
        x : α
        y : β
        h1 : Eq y b
        h2 : Not (Eq x a)
        ⊢ Iff (Membership.mem (Option.some a) x) (Membership.mem Option.none y)
      -/
    · simp only [mem_def, some.injEq, iff_false, reduceCtorEq] at *
      /-
        case neg
        α : Type u
        β : Type v
        γ : Type w
        δ : Type x
        inst✝² : DecidableEq α
        inst✝¹ : DecidableEq β
        inst✝ : DecidableEq γ
        a : α
        b : β
        x : α
        y : β
        h1 : Eq y b
        h2 : Not (Eq x a)
        ⊢ Not (Eq a x)
      -/
      exact Ne.symm h2
      /-
        🎉 no goals
      -/
      /-
        case pos
        α : Type u
        β : Type v
        γ : Type w
        δ : Type x
        inst✝² : DecidableEq α
        inst✝¹ : DecidableEq β
        inst✝ : DecidableEq γ
        a : α
        b : β
        x : α
        y : β
        h1 : Not (Eq y b)
        h✝ : Eq x a
        ⊢ Iff (Membership.mem Option.none x) (Membership.mem (Option.some b) y)
      -/
    · simp only [mem_def, some.injEq, false_iff, reduceCtorEq] at *
      /-
        case pos
        α : Type u
        β : Type v
        γ : Type w
        δ : Type x
        inst✝² : DecidableEq α
        inst✝¹ : DecidableEq β
        inst✝ : DecidableEq γ
        a : α
        b : β
        x : α
        y : β
        h1 : Not (Eq y b)
        h✝ : Eq x a
        ⊢ Not (Eq b y)
      -/
      exact Ne.symm h1
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        β : Type v
        γ : Type w
        δ : Type x
        inst✝² : DecidableEq α
        inst✝¹ : DecidableEq β
        inst✝ : DecidableEq γ
        a : α
        b : β
        x : α
        y : β
        h1 : Not (Eq y b)
        h✝ : Not (Eq x a)
        ⊢ Iff (Membership.mem Option.none x) (Membership.mem Option.none y)
      -/
    · simp
      /-
        🎉 no goals
      -/


theorem mem_single (a : α) (b : β) : b ∈ single a b a :=
  if_pos rfl


theorem mem_single_iff (a₁ a₂ : α) (b₁ b₂ : β) : b₁ ∈ single a₂ b₂ a₁ ↔ a₁ = a₂ ∧ b₁ = b₂ := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    a₁ a₂ : α
    b₁ b₂ : β
    ⊢ Iff (Membership.mem ((PEquiv.single a₂ b₂) a₁) b₁) (And (Eq a₁ a₂) (Eq b₁ b₂))
  -/
                                /-
                                  🎉 no goals
                                -/
  dsimp [single]; split_ifs <;> simp [*, eq_comm]
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem symm_single (a : α) (b : β) : (single a b).symm = single b a :=
  rfl


@[simp]
theorem single_apply (a : α) (b : β) : single a b a = some b :=
  if_pos rfl


theorem single_apply_of_ne {a₁ a₂ : α} (h : a₁ ≠ a₂) (b : β) : single a₁ b a₂ = none :=
  if_neg h.symm


theorem single_trans_of_mem (a : α) {b : β} {c : γ} {f : β ≃. γ} (h : c ∈ f b) :
    (single a b).trans f = single a c := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : DecidableEq α
    inst✝¹ : DecidableEq β
    inst✝ : DecidableEq γ
    a : α
    b : β
    c : γ
    f : PEquiv β γ
    h : Membership.mem (f b) c
    ⊢ Eq ((PEquiv.single a b).trans f) (PEquiv.single a c)
  -/
  ext
  /-
    case h.a
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : DecidableEq α
    inst✝¹ : DecidableEq β
    inst✝ : DecidableEq γ
    a : α
    b : β
    c : γ
    f : PEquiv β γ
    h : Membership.mem (f b) c
    x✝ : α
    a✝ : γ
    ⊢ Iff (Membership.mem (((PEquiv.single a b).trans f) x✝) a✝) (Membership.mem ( …
  -/
  dsimp [single, PEquiv.trans]
  /-
    case h.a
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : DecidableEq α
    inst✝¹ : DecidableEq β
    inst✝ : DecidableEq γ
    a : α
    b : β
    c : γ
    f : PEquiv β γ
    h : Membership.mem (f b) c
    x✝ : α
    a✝ : γ
    ⊢ Iff (Membership.mem ((ite (Eq x✝ a) (Option.some b) Option.none).bind ⇑f) a✝ …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp_all
                /-
                  🎉 no goals
                -/


theorem trans_single_of_mem {a : α} {b : β} (c : γ) {f : α ≃. β} (h : b ∈ f a) :
    f.trans (single b c) = single a c :=
  symm_injective <| single_trans_of_mem _ ((mem_iff_mem f).2 h)


@[simp]
theorem single_trans_single (a : α) (b : β) (c : γ) :
    (single a b).trans (single b c) = single a c :=
  single_trans_of_mem _ (mem_single _ _)


@[simp]
theorem single_subsingleton_eq_refl [Subsingleton α] (a b : α) : single a b = PEquiv.refl α := by
  /-
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Subsingleton α
    a b : α
    ⊢ Eq (PEquiv.single a b) (PEquiv.refl α)
  -/
  ext i j
  /-
    case h.a
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Subsingleton α
    a b i j : α
    ⊢ Iff (Membership.mem ((PEquiv.single a b) i) j) (Membership.mem ((PEquiv.refl …
  -/
  dsimp [single]
  /-
    case h.a
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Subsingleton α
    a b i j : α
    ⊢ Iff (Membership.mem (ite (Eq i a) (Option.some b) Option.none) j) (Membershi …
  -/
  rw [if_pos (Subsingleton.elim i a), Subsingleton.elim i j, Subsingleton.elim b j]
  /-
    🎉 no goals
  -/


theorem trans_single_of_eq_none {b : β} (c : γ) {f : δ ≃. β} (h : f.symm b = none) :
    f.trans (single b c) = ⊥ := by
  /-
    β : Type v
    γ : Type w
    δ : Type x
    inst✝¹ : DecidableEq β
    inst✝ : DecidableEq γ
    b : β
    c : γ
    f : PEquiv δ β
    h : Eq (f.symm b) Option.none
    ⊢ Eq (f.trans (PEquiv.single b c)) Bot.bot
  -/
  ext
  /-
    case h.a
    β : Type v
    γ : Type w
    δ : Type x
    inst✝¹ : DecidableEq β
    inst✝ : DecidableEq γ
    b : β
    c : γ
    f : PEquiv δ β
    h : Eq (f.symm b) Option.none
    x✝ : δ
    a✝ : γ
    ⊢ Iff (Membership.mem ((f.trans (PEquiv.single b c)) x✝) a✝) (Membership.mem ( …
  -/
  simp only [eq_none_iff_forall_not_mem, Option.mem_def, f.eq_some_iff] at h
  /-
    case h.a
    β : Type v
    γ : Type w
    δ : Type x
    inst✝¹ : DecidableEq β
    inst✝ : DecidableEq γ
    b : β
    c : γ
    f : PEquiv δ β
    x✝ : δ
    a✝ : γ
    h : ∀ (a : δ), Not (Eq (f a) (Option.some b))
    ⊢ Iff (Membership.mem ((f.trans (PEquiv.single b c)) x✝) a✝) (Membership.mem ( …
  -/
  dsimp [PEquiv.trans, single]
  /-
    case h.a
    β : Type v
    γ : Type w
    δ : Type x
    inst✝¹ : DecidableEq β
    inst✝ : DecidableEq γ
    b : β
    c : γ
    f : PEquiv δ β
    x✝ : δ
    a✝ : γ
    h : ∀ (a : δ), Not (Eq (f a) (Option.some b))
    ⊢ Iff (Membership.mem ((f x✝).bind fun x => ite (Eq x b) (Option.some c) Optio …
  -/
  simp only [mem_def, bind_eq_some, iff_false, not_exists, not_and, reduceCtorEq]
  /-
    case h.a
    β : Type v
    γ : Type w
    δ : Type x
    inst✝¹ : DecidableEq β
    inst✝ : DecidableEq γ
    b : β
    c : γ
    f : PEquiv δ β
    x✝ : δ
    a✝ : γ
    h : ∀ (a : δ), Not (Eq (f a) (Option.some b))
    ⊢ ∀ (x : β), Eq (f x✝) (Option.some x) → Not (Eq (ite (Eq x b) (Option.some c) …
  -/
  intros
  /-
    case h.a
    β : Type v
    γ : Type w
    δ : Type x
    inst✝¹ : DecidableEq β
    inst✝ : DecidableEq γ
    b : β
    c : γ
    f : PEquiv δ β
    x✝¹ : δ
    a✝¹ : γ
    h : ∀ (a : δ), Not (Eq (f a) (Option.some b))
    x✝ : β
    a✝ : Eq (f x✝¹) (Option.some x✝)
    ⊢ Not (Eq (ite (Eq x✝ b) (Option.some c) Option.none) (Option.some a✝¹))
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp_all
                /-
                  🎉 no goals
                -/


theorem single_trans_of_eq_none (a : α) {b : β} {f : β ≃. δ} (h : f b = none) :
    (single a b).trans f = ⊥ :=
  symm_injective <| trans_single_of_eq_none _ h


theorem single_trans_single_of_ne {b₁ b₂ : β} (h : b₁ ≠ b₂) (a : α) (c : γ) :
    (single a b₁).trans (single b₂ c) = ⊥ :=
  single_trans_of_eq_none _ (single_apply_of_ne h.symm _)


instance instPartialOrderPEquiv : PartialOrder (α ≃. β) where
  le f g := ∀ (a : α) (b : β), b ∈ f a → b ∈ g a
  le_refl _ _ _ := id
  le_trans _ _ _ fg gh a b := gh a b ∘ fg a b
  le_antisymm f g fg gf :=
    ext
      (by
        /-
          α : Type u
          β : Type v
          γ : Type w
          δ : Type x
          f g : PEquiv α β
          fg : LE.le f g
          gf : LE.le g f
          ⊢ ∀ (x : α), Eq (f x) (g x)
        -/
        intro a
        /-
          α : Type u
          β : Type v
          γ : Type w
          δ : Type x
          f g : PEquiv α β
          fg : LE.le f g
          gf : LE.le g f
          a : α
          ⊢ Eq (f a) (g a)
        -/
        rcases h : g a with _ | b
          /-
            case none
            α : Type u
            β : Type v
            γ : Type w
            δ : Type x
            f g : PEquiv α β
            fg : LE.le f g
            gf : LE.le g f
            a : α
            h : Eq (g a) Option.none
            ⊢ Eq (f a) Option.none
          -/
        · exact eq_none_iff_forall_not_mem.2 fun b hb => Option.not_mem_none b <| h ▸ fg a b hb
          /-
            🎉 no goals
          -/
          /-
            case some
            α : Type u
            β : Type v
            γ : Type w
            δ : Type x
            f g : PEquiv α β
            fg : LE.le f g
            gf : LE.le g f
            a : α
            b : β
            h : Eq (g a) (Option.some b)
            ⊢ Eq (f a) (Option.some b)
          -/
        · exact gf _ _ h)
          /-
            🎉 no goals
          -/


theorem le_def {f g : α ≃. β} : f ≤ g ↔ ∀ (a : α) (b : β), b ∈ f a → b ∈ g a :=
  Iff.rfl


instance : OrderBot (α ≃. β) :=
  { instBotPEquiv with bot_le := fun _ _ _ h => (not_mem_none _ h).elim }


instance [DecidableEq α] [DecidableEq β] : SemilatticeInf (α ≃. β) :=
  { instPartialOrderPEquiv with
    inf := fun f g =>
      { toFun := fun a => if f a = g a then f a else none
        invFun := fun b => if f.symm b = g.symm b then f.symm b else none
        inv := fun a b => by
          /-
            α : Type u
            β : Type v
            γ : Type w
            δ : Type x
            inst✝¹ : DecidableEq α
            inst✝ : DecidableEq β
            f g : PEquiv α β
            a : α
            b : β
            ⊢ Iff (Membership.mem ((fun b => ite (Eq (f.symm b) (g.symm b)) (f.symm b) Opt …
          -/
          have hf := @mem_iff_mem _ _ f a b
          /-
            α : Type u
            β : Type v
            γ : Type w
            δ : Type x
            inst✝¹ : DecidableEq α
            inst✝ : DecidableEq β
            f g : PEquiv α β
            a : α
            b : β
            hf : Iff (Membership.mem (f.symm b) a) (Membership.mem (f a) b)
            ⊢ Iff (Membership.mem ((fun b => ite (Eq (f.symm b) (g.symm b)) (f.symm b) Opt …
          -/
          have hg := @mem_iff_mem _ _ g a b
          /-
            α : Type u
            β : Type v
            γ : Type w
            δ : Type x
            inst✝¹ : DecidableEq α
            inst✝ : DecidableEq β
            f g : PEquiv α β
            a : α
            b : β
            hf : Iff (Membership.mem (f.symm b) a) (Membership.mem (f a) b)
            hg : Iff (Membership.mem (g.symm b) a) (Membership.mem (g a) b)
            ⊢ Iff (Membership.mem ((fun b => ite (Eq (f.symm b) (g.symm b)) (f.symm b) Opt …
          -/
          simp only [Option.mem_def] at *
          /-
            α : Type u
            β : Type v
            γ : Type w
            δ : Type x
            inst✝¹ : DecidableEq α
            inst✝ : DecidableEq β
            f g : PEquiv α β
            a : α
            b : β
            hf : Iff (Eq (f.symm b) (Option.some a)) (Eq (f a) (Option.some b))
            hg : Iff (Eq (g.symm b) (Option.some a)) (Eq (g a) (Option.some b))
            ⊢ Iff (Eq (ite (Eq (f.symm b) (g.symm b)) (f.symm b) Option.none) (Option.some …
          -/
                                      /-
                                        🎉 no goals
                                      -/
          split_ifs with h1 h2 h2 <;> try simp [hf]
                                      /-
                                        🎉 no goals
                                      -/
            /-
              case neg
              α : Type u
              β : Type v
              γ : Type w
              δ : Type x
              inst✝¹ : DecidableEq α
              inst✝ : DecidableEq β
              f g : PEquiv α β
              a : α
              b : β
              hf : Iff (Eq (f.symm b) (Option.some a)) (Eq (f a) (Option.some b))
              hg : Iff (Eq (g.symm b) (Option.some a)) (Eq (g a) (Option.some b))
              h1 : Eq (f.symm b) (g.symm b)
              h2 : Not (Eq (f a) (g a))
              ⊢ Not (Eq (f a) (Option.some b))
            -/
          · contrapose! h2
            /-
              case neg
              α : Type u
              β : Type v
              γ : Type w
              δ : Type x
              inst✝¹ : DecidableEq α
              inst✝ : DecidableEq β
              f g : PEquiv α β
              a : α
              b : β
              hf : Iff (Eq (f.symm b) (Option.some a)) (Eq (f a) (Option.some b))
              hg : Iff (Eq (g.symm b) (Option.some a)) (Eq (g a) (Option.some b))
              h1 : Eq (f.symm b) (g.symm b)
              h2 : Eq (f a) (Option.some b)
              ⊢ Eq (f a) (g a)
            -/
            rw [h2]
            /-
              case neg
              α : Type u
              β : Type v
              γ : Type w
              δ : Type x
              inst✝¹ : DecidableEq α
              inst✝ : DecidableEq β
              f g : PEquiv α β
              a : α
              b : β
              hf : Iff (Eq (f.symm b) (Option.some a)) (Eq (f a) (Option.some b))
              hg : Iff (Eq (g.symm b) (Option.some a)) (Eq (g a) (Option.some b))
              h1 : Eq (f.symm b) (g.symm b)
              h2 : Eq (f a) (Option.some b)
              ⊢ Eq (Option.some b) (g a)
            -/
            rw [← h1, hf, h2] at hg
            /-
              case neg
              α : Type u
              β : Type v
              γ : Type w
              δ : Type x
              inst✝¹ : DecidableEq α
              inst✝ : DecidableEq β
              f g : PEquiv α β
              a : α
              b : β
              hf : Iff (Eq (f.symm b) (Option.some a)) (Eq (f a) (Option.some b))
              hg : Iff (Eq (Option.some b) (Option.some b)) (Eq (g a) (Option.some b))
              h1 : Eq (f.symm b) (g.symm b)
              h2 : Eq (f a) (Option.some b)
              ⊢ Eq (Option.some b) (g a)
            -/
            simp only [mem_def, true_iff, eq_self_iff_true] at hg
            /-
              case neg
              α : Type u
              β : Type v
              γ : Type w
              δ : Type x
              inst✝¹ : DecidableEq α
              inst✝ : DecidableEq β
              f g : PEquiv α β
              a : α
              b : β
              hf : Iff (Eq (f.symm b) (Option.some a)) (Eq (f a) (Option.some b))
              h1 : Eq (f.symm b) (g.symm b)
              h2 : Eq (f a) (Option.some b)
              hg : Eq (g a) (Option.some b)
              ⊢ Eq (Option.some b) (g a)
            -/
            rw [hg]
            /-
              🎉 no goals
            -/
            /-
              case pos
              α : Type u
              β : Type v
              γ : Type w
              δ : Type x
              inst✝¹ : DecidableEq α
              inst✝ : DecidableEq β
              f g : PEquiv α β
              a : α
              b : β
              hf : Iff (Eq (f.symm b) (Option.some a)) (Eq (f a) (Option.some b))
              hg : Iff (Eq (g.symm b) (Option.some a)) (Eq (g a) (Option.some b))
              h1 : Not (Eq (f.symm b) (g.symm b))
              h2 : Eq (f a) (g a)
              ⊢ Not (Eq (f a) (Option.some b))
            -/
          · contrapose! h1
            /-
              case pos
              α : Type u
              β : Type v
              γ : Type w
              δ : Type x
              inst✝¹ : DecidableEq α
              inst✝ : DecidableEq β
              f g : PEquiv α β
              a : α
              b : β
              hf : Iff (Eq (f.symm b) (Option.some a)) (Eq (f a) (Option.some b))
              hg : Iff (Eq (g.symm b) (Option.some a)) (Eq (g a) (Option.some b))
              h2 : Eq (f a) (g a)
              h1 : Eq (f a) (Option.some b)
              ⊢ Eq (f.symm b) (g.symm b)
            -/
            rw [h1] at hf h2
            /-
              case pos
              α : Type u
              β : Type v
              γ : Type w
              δ : Type x
              inst✝¹ : DecidableEq α
              inst✝ : DecidableEq β
              f g : PEquiv α β
              a : α
              b : β
              hf : Iff (Eq (f.symm b) (Option.some a)) (Eq (Option.some b) (Option.some b))
              hg : Iff (Eq (g.symm b) (Option.some a)) (Eq (g a) (Option.some b))
              h2 : Eq (Option.some b) (g a)
              h1 : Eq (f a) (Option.some b)
              ⊢ Eq (f.symm b) (g.symm b)
            -/
            rw [← h2] at hg
            /-
              case pos
              α : Type u
              β : Type v
              γ : Type w
              δ : Type x
              inst✝¹ : DecidableEq α
              inst✝ : DecidableEq β
              f g : PEquiv α β
              a : α
              b : β
              hf : Iff (Eq (f.symm b) (Option.some a)) (Eq (Option.some b) (Option.some b))
              hg : Iff (Eq (g.symm b) (Option.some a)) (Eq (Option.some b) (Option.some b))
              h2 : Eq (Option.some b) (g a)
              h1 : Eq (f a) (Option.some b)
              ⊢ Eq (f.symm b) (g.symm b)
            -/
            simp only [iff_true] at hf hg
            /-
              case pos
              α : Type u
              β : Type v
              γ : Type w
              δ : Type x
              inst✝¹ : DecidableEq α
              inst✝ : DecidableEq β
              f g : PEquiv α β
              a : α
              b : β
              h2 : Eq (Option.some b) (g a)
              h1 : Eq (f a) (Option.some b)
              hf : Eq (f.symm b) (Option.some a)
              hg : Eq (g.symm b) (Option.some a)
              ⊢ Eq (f.symm b) (g.symm b)
            -/
            rw [hf, hg] }
            /-
              🎉 no goals
            -/
                                     /-
                                       α : Type u
                                       β : Type v
                                       γ : Type w
                                       δ : Type x
                                       inst✝¹ : DecidableEq α
                                       inst✝ : DecidableEq β
                                       x✝³ x✝² : PEquiv α β
                                       x✝¹ : α
                                       x✝ : β
                                       ⊢ Membership.mem (((fun f g => { toFun := fun a => ite (Eq (f a) (g a)) (f a)  …
                                     -/
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
    inf_le_left := fun _ _ _ _ => by simp only [coe_mk, mem_def]; split_ifs <;> simp [*]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
                                      /-
                                        α : Type u
                                        β : Type v
                                        γ : Type w
                                        δ : Type x
                                        inst✝¹ : DecidableEq α
                                        inst✝ : DecidableEq β
                                        x✝³ x✝² : PEquiv α β
                                        x✝¹ : α
                                        x✝ : β
                                        ⊢ Membership.mem (((fun f g => { toFun := fun a => ite (Eq (f a) (g a)) (f a)  …
                                      -/
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
    inf_le_right := fun _ _ _ _ => by simp only [coe_mk, mem_def]; split_ifs <;> simp [*]
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
    le_inf := fun f g h fg gh a b => by
      /-
        α : Type u
        β : Type v
        γ : Type w
        δ : Type x
        inst✝¹ : DecidableEq α
        inst✝ : DecidableEq β
        f g h : PEquiv α β
        fg : LE.le f g
        gh : LE.le f h
        a : α
        b : β
        ⊢ Membership.mem (f a) b → Membership.mem (((fun f g => { toFun := fun a => it …
      -/
      intro H
      /-
        α : Type u
        β : Type v
        γ : Type w
        δ : Type x
        inst✝¹ : DecidableEq α
        inst✝ : DecidableEq β
        f g h : PEquiv α β
        fg : LE.le f g
        gh : LE.le f h
        a : α
        b : β
        H : Membership.mem (f a) b
        ⊢ Membership.mem (((fun f g => { toFun := fun a => ite (Eq (f a) (g a)) (f a)  …
      -/
      have hf := fg a b H
      /-
        α : Type u
        β : Type v
        γ : Type w
        δ : Type x
        inst✝¹ : DecidableEq α
        inst✝ : DecidableEq β
        f g h : PEquiv α β
        fg : LE.le f g
        gh : LE.le f h
        a : α
        b : β
        H : Membership.mem (f a) b
        hf : Membership.mem (g a) b
        ⊢ Membership.mem (((fun f g => { toFun := fun a => ite (Eq (f a) (g a)) (f a)  …
      -/
      have hg := gh a b H
      /-
        α : Type u
        β : Type v
        γ : Type w
        δ : Type x
        inst✝¹ : DecidableEq α
        inst✝ : DecidableEq β
        f g h : PEquiv α β
        fg : LE.le f g
        gh : LE.le f h
        a : α
        b : β
        H : Membership.mem (f a) b
        hf : Membership.mem (g a) b
        hg : Membership.mem (h a) b
        ⊢ Membership.mem (((fun f g => { toFun := fun a => ite (Eq (f a) (g a)) (f a)  …
      -/
      simp only [Option.mem_def, PEquiv.coe_mk_apply] at *
      /-
        α : Type u
        β : Type v
        γ : Type w
        δ : Type x
        inst✝¹ : DecidableEq α
        inst✝ : DecidableEq β
        f g h : PEquiv α β
        fg : LE.le f g
        gh : LE.le f h
        a : α
        b : β
        H : Eq (f a) (Option.some b)
        hf : Eq (g a) (Option.some b)
        hg : Eq (h a) (Option.some b)
        ⊢ Eq (ite (Eq (g a) (h a)) (g a) Option.none) (Option.some b)
      -/
      rw [hf, hg, if_pos rfl] }
      /-
        🎉 no goals
      -/


/-- Turns an `Equiv` into a `PEquiv` of the whole type. -/
def toPEquiv (f : α ≃ β) : α ≃. β where
  toFun := some ∘ f
  invFun := some ∘ f.symm
            /-
              α : Type u_1
              β : Type u_2
              γ : Type u_3
              f : Equiv α β
              ⊢ ∀ (a : α) (b : β), Iff (Membership.mem (Function.comp Option.some (⇑f.symm)  …
            -/
  inv := by simp [Equiv.eq_symm_apply, eq_comm]
            /-
              🎉 no goals
            -/


@[simp]
theorem toPEquiv_refl : (Equiv.refl α).toPEquiv = PEquiv.refl α :=
  rfl


theorem toPEquiv_trans (f : α ≃ β) (g : β ≃ γ) :
    (f.trans g).toPEquiv = f.toPEquiv.trans g.toPEquiv :=
  rfl


theorem toPEquiv_symm (f : α ≃ β) : f.symm.toPEquiv = f.toPEquiv.symm :=
  rfl


theorem toPEquiv_apply (f : α ≃ β) (x : α) : f.toPEquiv x = some (f x) :=
  rfl


