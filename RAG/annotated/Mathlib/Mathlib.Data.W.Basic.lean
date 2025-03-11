/--
Given `β : α → Type*`, `WType β` is the type of finitely branching trees where nodes are labeled by
elements of `α` and the children of a node labeled `a` are indexed by elements of `β a`.
-/
inductive WType {α : Type*} (β : α → Type*)
  | mk (a : α) (f : β a → WType β) : WType β


instance : Inhabited (WType fun _ : Unit => Empty) :=
  ⟨WType.mk Unit.unit Empty.elim⟩


/-- The canonical map to the corresponding sigma type, returning the label of a node as an
  element `a` of `α`, and the children of the node as a function `β a → WType β`. -/
def toSigma : WType β → Σa : α, β a → WType β
  | ⟨a, f⟩ => ⟨a, f⟩


/-- The canonical map from the sigma type into a `WType`. Given a node `a : α`, and
  its children as a function `β a → WType β`, return the corresponding tree. -/
def ofSigma : (Σa : α, β a → WType β) → WType β
  | ⟨a, f⟩ => WType.mk a f


@[simp]
theorem ofSigma_toSigma : ∀ w : WType β, ofSigma (toSigma w) = w
  | ⟨_, _⟩ => rfl


@[simp]
theorem toSigma_ofSigma : ∀ s : Σa : α, β a → WType β, toSigma (ofSigma s) = s
  | ⟨_, _⟩ => rfl


/-- The canonical bijection with the sigma type, showing that `WType` is a fixed point of
  the polynomial functor `X ↦ Σ a : α, β a → X`. -/
@[simps]
def equivSigma : WType β ≃ Σa : α, β a → WType β where
  toFun := toSigma
  invFun := ofSigma
  left_inv := ofSigma_toSigma
  right_inv := toSigma_ofSigma


/-- The canonical map from `WType β` into any type `γ` given a map `(Σ a : α, β a → γ) → γ`. -/
def elim (γ : Type*) (fγ : (Σa : α, β a → γ) → γ) : WType β → γ
  | ⟨a, f⟩ => fγ ⟨a, fun b => elim γ fγ (f b)⟩


theorem elim_injective (γ : Type*) (fγ : (Σa : α, β a → γ) → γ)
    (fγ_injective : Function.Injective fγ) : Function.Injective (elim γ fγ)
  | ⟨a₁, f₁⟩, ⟨a₂, f₂⟩, h => by
    /-
      α : Type u_1
      β : α → Type u_2
      γ : Type u_3
      fγ : (Sigma fun a => β a → γ) → γ
      fγ_injective : Function.Injective fγ
      a₁ : α
      f₁ : β a₁ → WType β
      a₂ : α
      f₂ : β a₂ → WType β
      h : Eq (WType.elim γ fγ (WType.mk a₁ f₁)) (WType.elim γ fγ (WType.mk a₂ f₂))
      ⊢ Eq (WType.mk a₁ f₁) (WType.mk a₂ f₂)
    -/
    obtain ⟨rfl, h⟩ := Sigma.mk.inj_iff.mp (fγ_injective h)
    /-
      case intro
      α : Type u_1
      β : α → Type u_2
      γ : Type u_3
      fγ : (Sigma fun a => β a → γ) → γ
      fγ_injective : Function.Injective fγ
      a₁ : α
      f₁ f₂ : β a₁ → WType β
      h✝ : Eq (WType.elim γ fγ (WType.mk a₁ f₁)) (WType.elim γ fγ (WType.mk a₁ f₂))
      h : HEq (fun b => WType.elim γ fγ (f₁ b)) fun b => WType.elim γ fγ (f₂ b)
      ⊢ Eq (WType.mk a₁ f₁) (WType.mk a₁ f₂)
    -/
    congr with x
    /-
      case intro.e_f.h
      α : Type u_1
      β : α → Type u_2
      γ : Type u_3
      fγ : (Sigma fun a => β a → γ) → γ
      fγ_injective : Function.Injective fγ
      a₁ : α
      f₁ f₂ : β a₁ → WType β
      h✝ : Eq (WType.elim γ fγ (WType.mk a₁ f₁)) (WType.elim γ fγ (WType.mk a₁ f₂))
      h : HEq (fun b => WType.elim γ fγ (f₁ b)) fun b => WType.elim γ fγ (f₂ b)
      x : β a₁
      ⊢ Eq (f₁ x) (f₂ x)
    -/
    exact elim_injective γ fγ fγ_injective (congr_fun (eq_of_heq h) x : _)
    /-
      🎉 no goals
    -/


instance [hα : IsEmpty α] : IsEmpty (WType β) :=
  ⟨fun w => WType.recOn w (IsEmpty.elim hα)⟩


theorem infinite_of_nonempty_of_isEmpty (a b : α) [ha : Nonempty (β a)] [he : IsEmpty (β b)] :
    Infinite (WType β) :=
  ⟨by
    /-
      α : Type u_1
      β : α → Type u_2
      a b : α
      ha : Nonempty (β a)
      he : IsEmpty (β b)
      ⊢ Not (Finite (WType β))
    -/
    intro hf
    /-
      α : Type u_1
      β : α → Type u_2
      a b : α
      ha : Nonempty (β a)
      he : IsEmpty (β b)
      hf : Finite (WType β)
      ⊢ False
    -/
    have hba : b ≠ a := fun h => ha.elim (IsEmpty.elim' (show IsEmpty (β a) from h ▸ he))
    refine
      not_injective_infinite_finite
        (fun n : ℕ =>
          show WType β from Nat.recOn n ⟨b, IsEmpty.elim' he⟩ fun _ ih => ⟨a, fun _ => ih⟩)
        ?_
    /-
      α : Type u_1
      β : α → Type u_2
      a b : α
      ha : Nonempty (β a)
      he : IsEmpty (β b)
      hf : Finite (WType β)
      hba : Ne b a
      ⊢ Function.Injective fun n => letFun (Nat.recOn n (WType.mk b he.elim') fun x  …
    -/
    intro n m h
    /-
      α : Type u_1
      β : α → Type u_2
      a b : α
      ha : Nonempty (β a)
      he : IsEmpty (β b)
      hf : Finite (WType β)
      hba : Ne b a
      n m : Nat
      h : Eq ((fun n => letFun (Nat.recOn n (WType.mk b he.elim') fun x ih => WType. …
      ⊢ Eq n m
    -/
    induction' n with n ih generalizing m
      /-
        case zero
        α : Type u_1
        β : α → Type u_2
        a b : α
        ha : Nonempty (β a)
        he : IsEmpty (β b)
        hf : Finite (WType β)
        hba : Ne b a
        m : Nat
        h : Eq ((fun n => letFun (Nat.recOn n (WType.mk b he.elim') fun x ih => WType. …
        ⊢ Eq 0 m
      -/
                          /-
                            🎉 no goals
                          -/
    · cases' m with m <;> simp_all
                          /-
                            🎉 no goals
                          -/
      /-
        case succ
        α : Type u_1
        β : α → Type u_2
        a b : α
        ha : Nonempty (β a)
        he : IsEmpty (β b)
        hf : Finite (WType β)
        hba : Ne b a
        n : Nat
        ih : ∀ ⦃m : Nat⦄, Eq ((fun n => letFun (Nat.recOn n (WType.mk b he.elim') fun  …
        m : Nat
        h : Eq ((fun n => letFun (Nat.recOn n (WType.mk b he.elim') fun x ih => WType. …
        ⊢ Eq (HAdd.hAdd n 1) m
      -/
    · cases' m with m
        /-
          case succ.zero
          α : Type u_1
          β : α → Type u_2
          a b : α
          ha : Nonempty (β a)
          he : IsEmpty (β b)
          hf : Finite (WType β)
          hba : Ne b a
          n : Nat
          ih : ∀ ⦃m : Nat⦄, Eq ((fun n => letFun (Nat.recOn n (WType.mk b he.elim') fun  …
          h : Eq ((fun n => letFun (Nat.recOn n (WType.mk b he.elim') fun x ih => WType. …
          ⊢ Eq (HAdd.hAdd n 1) 0
        -/
      · simp_all
        /-
          🎉 no goals
        -/
        /-
          case succ.succ
          α : Type u_1
          β : α → Type u_2
          a b : α
          ha : Nonempty (β a)
          he : IsEmpty (β b)
          hf : Finite (WType β)
          hba : Ne b a
          n : Nat
          ih : ∀ ⦃m : Nat⦄, Eq ((fun n => letFun (Nat.recOn n (WType.mk b he.elim') fun  …
          m : Nat
          h : Eq ((fun n => letFun (Nat.recOn n (WType.mk b he.elim') fun x ih => WType. …
          ⊢ Eq (HAdd.hAdd n 1) (HAdd.hAdd m 1)
        -/
      · refine congr_arg Nat.succ (ih ?_)
        /-
          case succ.succ
          α : Type u_1
          β : α → Type u_2
          a b : α
          ha : Nonempty (β a)
          he : IsEmpty (β b)
          hf : Finite (WType β)
          hba : Ne b a
          n : Nat
          ih : ∀ ⦃m : Nat⦄, Eq ((fun n => letFun (Nat.recOn n (WType.mk b he.elim') fun  …
          m : Nat
          h : Eq ((fun n => letFun (Nat.recOn n (WType.mk b he.elim') fun x ih => WType. …
          ⊢ Eq ((fun n => letFun (Nat.recOn n (WType.mk b he.elim') fun x ih => WType.mk …
        -/
        simp_all [funext_iff]⟩
        /-
          🎉 no goals
        -/


/-- The depth of a finitely branching tree. -/
def depth : WType β → ℕ
  | ⟨_, f⟩ => (Finset.sup Finset.univ fun n => depth (f n)) + 1


theorem depth_pos (t : WType β) : 0 < t.depth := by
  /-
    α : Type u_1
    β : α → Type u_2
    inst✝ : (a : α) → Fintype (β a)
    t : WType β
    ⊢ LT.lt 0 t.depth
  -/
  cases t
  /-
    case mk
    α : Type u_1
    β : α → Type u_2
    inst✝ : (a : α) → Fintype (β a)
    a✝ : α
    f✝ : β a✝ → WType β
    ⊢ LT.lt 0 (WType.mk a✝ f✝).depth
  -/
  apply Nat.succ_pos
  /-
    🎉 no goals
  -/


theorem depth_lt_depth_mk (a : α) (f : β a → WType β) (i : β a) : depth (f i) < depth ⟨a, f⟩ :=
  Nat.lt_succ_of_le (Finset.le_sup (f := (depth <| f ·)) (Finset.mem_univ i))

/-
Show that W types are encodable when `α` is an encodable fintype and for every `a : α`, `β a` is
encodable.

We define an auxiliary type `WType' β n` of trees of depth at most `n`, and then we show by
induction on `n` that these are all encodable. These auxiliary constructions are not interesting in
and of themselves, so we mark them as `private`.
-/

private abbrev WType' {α : Type*} (β : α → Type*) [∀ a : α, Fintype (β a)]
    [∀ a : α, Encodable (β a)] (n : ℕ) :=
  { t : WType β // t.depth ≤ n }


private def encodable_zero : Encodable (WType' β 0) :=
  let f : WType' β 0 → Empty := fun ⟨_, h⟩ => False.elim <| not_lt_of_ge h (WType.depth_pos _)
  let finv : Empty → WType' β 0 := by
    /-
      α : Type u_1
      β : α → Type u_2
      inst✝¹ : (a : α) → Fintype (β a)
      inst✝ : (a : α) → Encodable (β a)
      f : WType.WType' β 0 → Empty := fun x => WType.encodable_zero.match_1 (fun x = …
      ⊢ Empty → WType.WType' β 0
    -/
    intro x
    /-
      α : Type u_1
      β : α → Type u_2
      inst✝¹ : (a : α) → Fintype (β a)
      inst✝ : (a : α) → Encodable (β a)
      f : WType.WType' β 0 → Empty := fun x => WType.encodable_zero.match_1 (fun x = …
      x : Empty
      ⊢ WType.WType' β 0
    -/
    cases x
    /-
      🎉 no goals
    -/
  have : ∀ x, finv (f x) = x := fun ⟨_, h⟩ => False.elim <| not_lt_of_ge h (WType.depth_pos _)
  Encodable.ofLeftInverse f finv this


private def f (n : ℕ) : WType' β (n + 1) → Σa : α, β a → WType' β n
  | ⟨t, h⟩ => by
    /-
      α : Type u_1
      β : α → Type u_2
      inst✝¹ : (a : α) → Fintype (β a)
      inst✝ : (a : α) → Encodable (β a)
      n : Nat
      t : WType β
      h : LE.le t.depth (HAdd.hAdd n 1)
      ⊢ Sigma fun a => β a → WType.WType' β n
    -/
    cases' t with a f
    have h₀ : ∀ i : β a, WType.depth (f i) ≤ n := fun i =>
      Nat.le_of_lt_succ (lt_of_lt_of_le (WType.depth_lt_depth_mk a f i) h)
    /-
      case mk
      α : Type u_1
      β : α → Type u_2
      inst✝¹ : (a : α) → Fintype (β a)
      inst✝ : (a : α) → Encodable (β a)
      n : Nat
      a : α
      f : β a → WType β
      h : LE.le (WType.mk a f).depth (HAdd.hAdd n 1)
      h₀ : ∀ (i : β a), LE.le (f i).depth n
      ⊢ Sigma fun a => β a → WType.WType' β n
    -/
    exact ⟨a, fun i : β a => ⟨f i, h₀ i⟩⟩
    /-
      🎉 no goals
    -/


private def finv (n : ℕ) : (Σa : α, β a → WType' β n) → WType' β (n + 1)
  | ⟨a, f⟩ =>
    let f' := fun i : β a => (f i).val
    have : WType.depth ⟨a, f'⟩ ≤ n + 1 := Nat.add_le_add_right (Finset.sup_le fun b _ => (f b).2) 1
    ⟨⟨a, f'⟩, this⟩


private def encodable_succ (n : Nat) (_ : Encodable (WType' β n)) : Encodable (WType' β (n + 1)) :=
  Encodable.ofLeftInverse (f n) (finv n)
    (by
      /-
        α : Type u_1
        β : α → Type u_2
        inst✝² : (a : α) → Fintype (β a)
        inst✝¹ : (a : α) → Encodable (β a)
        inst✝ : Encodable α
        n : Nat
        x✝ : Encodable (WType.WType' β n)
        ⊢ ∀ (b : WType.WType' β (HAdd.hAdd n 1)), Eq (WType.finv n (WType.f n b)) b
      -/
      rintro ⟨⟨_, _⟩, _⟩
      /-
        case mk.mk
        α : Type u_1
        β : α → Type u_2
        inst✝² : (a : α) → Fintype (β a)
        inst✝¹ : (a : α) → Encodable (β a)
        inst✝ : Encodable α
        n : Nat
        x✝ : Encodable (WType.WType' β n)
        a✝ : α
        f✝ : β a✝ → WType β
        property✝ : LE.le (WType.mk a✝ f✝).depth (HAdd.hAdd n 1)
        ⊢ Eq (WType.finv n (WType.f n ⟨WType.mk a✝ f✝, property✝⟩)) ⟨WType.mk a✝ f✝, p …
      -/
      rfl)
      /-
        🎉 no goals
      -/


/-- `WType` is encodable when `α` is an encodable fintype and for every `a : α`, `β a` is
encodable. -/
instance : Encodable (WType β) := by
  /-
    α : Type u_1
    β : α → Type u_2
    inst✝² : (a : α) → Fintype (β a)
    inst✝¹ : (a : α) → Encodable (β a)
    inst✝ : Encodable α
    ⊢ Encodable (WType β)
  -/
  haveI h' : ∀ n, Encodable (WType' β n) := fun n => Nat.rec encodable_zero encodable_succ n
  /-
    α : Type u_1
    β : α → Type u_2
    inst✝² : (a : α) → Fintype (β a)
    inst✝¹ : (a : α) → Encodable (β a)
    inst✝ : Encodable α
    h' : (n : Nat) → Encodable (WType.WType' β n)
    ⊢ Encodable (WType β)
  -/
  let f : WType β → Σn, WType' β n := fun t => ⟨t.depth, ⟨t, le_rfl⟩⟩
  /-
    α : Type u_1
    β : α → Type u_2
    inst✝² : (a : α) → Fintype (β a)
    inst✝¹ : (a : α) → Encodable (β a)
    inst✝ : Encodable α
    h' : (n : Nat) → Encodable (WType.WType' β n)
    f : WType β → Sigma fun n => WType.WType' β n := fun t => ⟨t.depth, ⟨t, ⋯⟩⟩
    ⊢ Encodable (WType β)
  -/
  let finv : (Σn, WType' β n) → WType β := fun p => p.2.1
  /-
    α : Type u_1
    β : α → Type u_2
    inst✝² : (a : α) → Fintype (β a)
    inst✝¹ : (a : α) → Encodable (β a)
    inst✝ : Encodable α
    h' : (n : Nat) → Encodable (WType.WType' β n)
    f : WType β → Sigma fun n => WType.WType' β n := fun t => ⟨t.depth, ⟨t, ⋯⟩⟩
    finv : (Sigma fun n => WType.WType' β n) → WType β := fun p => ↑p.snd
    ⊢ Encodable (WType β)
  -/
  have : ∀ t, finv (f t) = t := fun t => rfl
  /-
    α : Type u_1
    β : α → Type u_2
    inst✝² : (a : α) → Fintype (β a)
    inst✝¹ : (a : α) → Encodable (β a)
    inst✝ : Encodable α
    h' : (n : Nat) → Encodable (WType.WType' β n)
    f : WType β → Sigma fun n => WType.WType' β n := fun t => ⟨t.depth, ⟨t, ⋯⟩⟩
    finv : (Sigma fun n => WType.WType' β n) → WType β := fun p => ↑p.snd
    this : ∀ (t : WType β), Eq (finv (f t)) t
    ⊢ Encodable (WType β)
  -/
  exact Encodable.ofLeftInverse f finv this
  /-
    🎉 no goals
  -/


