/--
This theorem exists because plausible does not have access to dlookup but
mathlib has all the theory for it and wants to use it. We probably want to
bring these two together at some point.
-/
private theorem apply_eq_dlookup (m : List (Σ _ : α, β)) (y : β) (x : α) :
     (withDefault m y).apply x = (m.dlookup x).getD y := by
  /-
    α : Type u
    β : Type v
    inst✝ : DecidableEq α
    m : List (Sigma fun x => β)
    y : β
    x : α
    ⊢ Eq ((Plausible.TotalFunction.withDefault m y).apply x) ((List.dlookup x m).g …
  -/
  dsimp only [apply]
  /-
    α : Type u
    β : Type v
    inst✝ : DecidableEq α
    m : List (Sigma fun x => β)
    y : β
    x : α
    ⊢ Eq ((Option.map Sigma.snd (List.find? (fun x_1 => Decidable.decide (Eq x_1.f …
  -/
  congr 1
  induction m with
  | nil => simp
  | cons p m ih =>
    rcases p with ⟨fst, snd⟩
    by_cases heq : fst = x
    · simp [heq]
    · rw [List.dlookup_cons_ne]
      · simp [heq, ih]
      · symm
        simp [heq]


/-- Map a total_function to one whose default value is zero so that it represents a finsupp. -/
@[simp]
def zeroDefault : TotalFunction α β → TotalFunction α β
  | .withDefault A _ => .withDefault A 0


/-- The support of a zero default `TotalFunction`. -/
@[simp]
def zeroDefaultSupp : TotalFunction α β → Finset α
  | .withDefault A _ =>
    List.toFinset <| (A.dedupKeys.filter fun ab => Sigma.snd ab ≠ 0).map Sigma.fst


/-- Create a finitely supported function from a total function by taking the default value to
zero. -/
def applyFinsupp (tf : TotalFunction α β) : α →₀ β where
  support := zeroDefaultSupp tf
  toFun := tf.zeroDefault.apply
  mem_support_toFun := by
    /-
      α : Type u
      β : Type v
      inst✝² : DecidableEq α
      inst✝¹ : Zero β
      inst✝ : DecidableEq β
      tf : Plausible.TotalFunction α β
      ⊢ ∀ (a : α), Iff (Membership.mem tf.zeroDefaultSupp a) (Ne (tf.zeroDefault.app …
    -/
    intro a
    /-
      α : Type u
      β : Type v
      inst✝² : DecidableEq α
      inst✝¹ : Zero β
      inst✝ : DecidableEq β
      tf : Plausible.TotalFunction α β
      a : α
      ⊢ Iff (Membership.mem tf.zeroDefaultSupp a) (Ne (tf.zeroDefault.apply a) 0)
    -/
    rcases tf with ⟨A, y⟩
    simp only [zeroDefaultSupp, List.mem_map, List.mem_filter, exists_and_right,
      List.mem_toFinset, exists_eq_right, Sigma.exists, Ne, zeroDefault]
    /-
      case withDefault
      α : Type u
      β : Type v
      inst✝² : DecidableEq α
      inst✝¹ : Zero β
      inst✝ : DecidableEq β
      a : α
      A : List (Sigma fun x => β)
      y : β
      ⊢ Iff (Exists fun x => And (Membership.mem A.dedupKeys ⟨a, x⟩) (Eq (Decidable. …
    -/
    rw [apply_eq_dlookup]
    /-
      case withDefault
      α : Type u
      β : Type v
      inst✝² : DecidableEq α
      inst✝¹ : Zero β
      inst✝ : DecidableEq β
      a : α
      A : List (Sigma fun x => β)
      y : β
      ⊢ Iff (Exists fun x => And (Membership.mem A.dedupKeys ⟨a, x⟩) (Eq (Decidable. …
    -/
    constructor
      /-
        case withDefault.mp
        α : Type u
        β : Type v
        inst✝² : DecidableEq α
        inst✝¹ : Zero β
        inst✝ : DecidableEq β
        a : α
        A : List (Sigma fun x => β)
        y : β
        ⊢ (Exists fun x => And (Membership.mem A.dedupKeys ⟨a, x⟩) (Eq (Decidable.deci …
      -/
    · rintro ⟨od, hval, hod⟩
      /-
        case withDefault.mp.intro.intro
        α : Type u
        β : Type v
        inst✝² : DecidableEq α
        inst✝¹ : Zero β
        inst✝ : DecidableEq β
        a : α
        A : List (Sigma fun x => β)
        y od : β
        hval : Membership.mem A.dedupKeys ⟨a, od⟩
        hod : Eq (Decidable.decide (Not (Eq od 0))) Bool.true
        ⊢ Not (Eq ((List.dlookup a A).getD 0) 0)
      -/
      have := List.mem_dlookup (List.nodupKeys_dedupKeys A) hval
      /-
        case withDefault.mp.intro.intro
        α : Type u
        β : Type v
        inst✝² : DecidableEq α
        inst✝¹ : Zero β
        inst✝ : DecidableEq β
        a : α
        A : List (Sigma fun x => β)
        y od : β
        hval : Membership.mem A.dedupKeys ⟨a, od⟩
        hod : Eq (Decidable.decide (Not (Eq od 0))) Bool.true
        this : Membership.mem (List.dlookup a A.dedupKeys) od
        ⊢ Not (Eq ((List.dlookup a A).getD 0) 0)
      -/
      rw [(_ : List.dlookup a A = od)]
        /-
          case withDefault.mp.intro.intro
          α : Type u
          β : Type v
          inst✝² : DecidableEq α
          inst✝¹ : Zero β
          inst✝ : DecidableEq β
          a : α
          A : List (Sigma fun x => β)
          y od : β
          hval : Membership.mem A.dedupKeys ⟨a, od⟩
          hod : Eq (Decidable.decide (Not (Eq od 0))) Bool.true
          this : Membership.mem (List.dlookup a A.dedupKeys) od
          ⊢ Not (Eq ((Option.some od).getD 0) 0)
        -/
      · simpa using hod
        /-
          🎉 no goals
        -/
        /-
          α : Type u
          β : Type v
          inst✝² : DecidableEq α
          inst✝¹ : Zero β
          inst✝ : DecidableEq β
          a : α
          A : List (Sigma fun x => β)
          y od : β
          hval : Membership.mem A.dedupKeys ⟨a, od⟩
          hod : Eq (Decidable.decide (Not (Eq od 0))) Bool.true
          this : Membership.mem (List.dlookup a A.dedupKeys) od
          ⊢ Eq (List.dlookup a A) (Option.some od)
        -/
      · simpa [List.dlookup_dedupKeys]
        /-
          🎉 no goals
        -/
      /-
        case withDefault.mpr
        α : Type u
        β : Type v
        inst✝² : DecidableEq α
        inst✝¹ : Zero β
        inst✝ : DecidableEq β
        a : α
        A : List (Sigma fun x => β)
        y : β
        ⊢ Not (Eq ((List.dlookup a A).getD 0) 0) → Exists fun x => And (Membership.mem …
      -/
    · intro h
      /-
        case withDefault.mpr
        α : Type u
        β : Type v
        inst✝² : DecidableEq α
        inst✝¹ : Zero β
        inst✝ : DecidableEq β
        a : α
        A : List (Sigma fun x => β)
        y : β
        h : Not (Eq ((List.dlookup a A).getD 0) 0)
        ⊢ Exists fun x => And (Membership.mem A.dedupKeys ⟨a, x⟩) (Eq (Decidable.decid …
      -/
      use (A.dlookup a).getD (0 : β)
      /-
        case h
        α : Type u
        β : Type v
        inst✝² : DecidableEq α
        inst✝¹ : Zero β
        inst✝ : DecidableEq β
        a : α
        A : List (Sigma fun x => β)
        y : β
        h : Not (Eq ((List.dlookup a A).getD 0) 0)
        ⊢ And (Membership.mem A.dedupKeys ⟨a, (List.dlookup a A).getD 0⟩) (Eq (Decidab …
      -/
      rw [← List.dlookup_dedupKeys] at h ⊢
      /-
        case h
        α : Type u
        β : Type v
        inst✝² : DecidableEq α
        inst✝¹ : Zero β
        inst✝ : DecidableEq β
        a : α
        A : List (Sigma fun x => β)
        y : β
        h : Not (Eq ((List.dlookup a A.dedupKeys).getD 0) 0)
        ⊢ And (Membership.mem A.dedupKeys ⟨a, (List.dlookup a A.dedupKeys).getD 0⟩) (E …
      -/
      simp only [h, ← List.mem_dlookup_iff A.nodupKeys_dedupKeys, not_false_iff, Option.mem_def]
      /-
        case h
        α : Type u
        β : Type v
        inst✝² : DecidableEq α
        inst✝¹ : Zero β
        inst✝ : DecidableEq β
        a : α
        A : List (Sigma fun x => β)
        y : β
        h : Not (Eq ((List.dlookup a A.dedupKeys).getD 0) 0)
        ⊢ And (Eq (List.dlookup a A.dedupKeys) (Option.some ((List.dlookup a A.dedupKe …
      -/
      cases haA : List.dlookup a A.dedupKeys
        /-
          case h.none
          α : Type u
          β : Type v
          inst✝² : DecidableEq α
          inst✝¹ : Zero β
          inst✝ : DecidableEq β
          a : α
          A : List (Sigma fun x => β)
          y : β
          h : Not (Eq ((List.dlookup a A.dedupKeys).getD 0) 0)
          haA : Eq (List.dlookup a A.dedupKeys) Option.none
          ⊢ And (Eq Option.none (Option.some (Option.none.getD 0))) (Eq (Decidable.decid …
        -/
      · simp [haA] at h
        /-
          🎉 no goals
        -/
        /-
          case h.some
          α : Type u
          β : Type v
          inst✝² : DecidableEq α
          inst✝¹ : Zero β
          inst✝ : DecidableEq β
          a : α
          A : List (Sigma fun x => β)
          y : β
          h : Not (Eq ((List.dlookup a A.dedupKeys).getD 0) 0)
          val✝ : β
          haA : Eq (List.dlookup a A.dedupKeys) (Option.some val✝)
          ⊢ And (Eq (Option.some val✝) (Option.some ((Option.some val✝).getD 0))) (Eq (D …
        -/
      · simp
        /-
          🎉 no goals
        -/


instance Finsupp.sampleableExt : SampleableExt (α →₀ β) where
  proxy := TotalFunction α (SampleableExt.proxy β)
  interp := fun f => (f.comp SampleableExt.interp).applyFinsupp
  sample := SampleableExt.sample (α := α → β)
  -- note: no way of shrinking the domain without an inverse to `interp`
  shrink := { shrink := letI : Shrinkable α := {}; TotalFunction.shrink }

-- TODO: support a non-constant codomain type

instance DFinsupp.sampleableExt : SampleableExt (Π₀ _ : α, β) where
  proxy := TotalFunction α (SampleableExt.proxy β)
  interp := fun f => (f.comp SampleableExt.interp).applyFinsupp.toDFinsupp
  sample := SampleableExt.sample (α := α → β)
  -- note: no way of shrinking the domain without an inverse to `interp`
  shrink := { shrink := letI : Shrinkable α := {}; TotalFunction.shrink }


/-- Data structure specifying a total function using a list of pairs
and a default value returned when the input is not in the domain of
the partial function.

`mapToSelf f` encodes `x ↦ f x` when `x ∈ f` and `x ↦ x`,
i.e. `x` to itself, otherwise.

We use `Σ` to encode mappings instead of `×` because we
rely on the association list API defined in `Mathlib/Data/List/Sigma.lean`.
-/
inductive InjectiveFunction (α : Type u) : Type u
  | mapToSelf (xs : List (Σ _ : α, α)) :
      xs.map Sigma.fst ~ xs.map Sigma.snd → List.Nodup (xs.map Sigma.snd) → InjectiveFunction α


instance : Inhabited (InjectiveFunction α) :=
  ⟨⟨[], List.Perm.nil, List.nodup_nil⟩⟩


/-- Apply a total function to an argument. -/
def apply [DecidableEq α] : InjectiveFunction α → α → α
  | InjectiveFunction.mapToSelf m _ _, x => (m.dlookup x).getD x


/-- Produce a string for a given `InjectiveFunction`.
The output is of the form `[x₀ ↦ f x₀, .. xₙ ↦ f xₙ, x ↦ x]`.
Unlike for `TotalFunction`, the default value is not a constant
but the identity function.
-/
protected def repr [Repr α] : InjectiveFunction α → String
  | InjectiveFunction.mapToSelf m _ _ => s! "[{TotalFunction.reprAux m}x ↦ x]"


instance (α : Type u) [Repr α] : Repr (InjectiveFunction α) where
  reprPrec f _p := InjectiveFunction.repr f


/-- Interpret a list of pairs as a total function, defaulting to
the identity function when no entries are found for a given function -/
def List.applyId [DecidableEq α] (xs : List (α × α)) (x : α) : α :=
  ((xs.map Prod.toSigma).dlookup x).getD x


@[simp]
theorem List.applyId_cons [DecidableEq α] (xs : List (α × α)) (x y z : α) :
    List.applyId ((y, z)::xs) x = if y = x then z else List.applyId xs x := by
  /-
    α : Type u
    inst✝ : DecidableEq α
    xs : List (Prod α α)
    x y z : α
    ⊢ Eq (Plausible.InjectiveFunction.List.applyId (List.cons { fst := y, snd := z …
  -/
  simp only [List.applyId, List.dlookup, eq_rec_constant, Prod.toSigma, List.map]
  /-
    α : Type u
    inst✝ : DecidableEq α
    xs : List (Prod α α)
    x y z : α
    ⊢ Eq ((dite (Eq y x) (fun h => Option.some z) fun h => List.dlookup x (List.ma …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> rfl
                /-
                  🎉 no goals
                -/


theorem List.applyId_zip_eq [DecidableEq α] {xs ys : List α} (h₀ : List.Nodup xs)
    (h₁ : xs.length = ys.length) (x y : α) (i : ℕ) (h₂ : xs[i]? = some x) :
    List.applyId.{u} (xs.zip ys) x = y ↔ ys[i]? = some y := by
  induction xs generalizing ys i with
  | nil => cases h₂
  | cons x' xs xs_ih =>
    cases i
    · simp only [length_cons, lt_add_iff_pos_left, add_pos_iff, Nat.lt_add_one, or_true,
        getElem?_eq_getElem, getElem_cons_zero, Option.some.injEq] at h₂
      subst h₂
      cases ys
      · cases h₁
      · simp
    · cases ys
      · cases h₁
      · cases' h₀ with _ _ h₀ h₁
        simp only [getElem?_cons_succ, zip_cons_cons, applyId_cons] at h₂ ⊢
        rw [if_neg]
        · apply xs_ih <;> solve_by_elim [Nat.succ.inj]
        · apply h₀; apply List.mem_of_getElem? h₂


theorem applyId_mem_iff [DecidableEq α] {xs ys : List α} (h₀ : List.Nodup xs) (h₁ : xs ~ ys)
    (x : α) : List.applyId.{u} (xs.zip ys) x ∈ ys ↔ x ∈ xs := by
  /-
    α : Type u
    inst✝ : DecidableEq α
    xs ys : List α
    h₀ : xs.Nodup
    h₁ : xs.Perm ys
    x : α
    ⊢ Iff (Membership.mem ys (Plausible.InjectiveFunction.List.applyId (xs.zip ys) …
  -/
  simp only [List.applyId]
  cases h₃ : List.dlookup x (List.map Prod.toSigma (xs.zip ys)) with
  | none =>
    dsimp [Option.getD]
    rw [h₁.mem_iff]
  | some val =>
    have h₂ : ys.Nodup := h₁.nodup_iff.1 h₀
    replace h₁ : xs.length = ys.length := h₁.length_eq
    dsimp
    induction xs generalizing ys with
    | nil => contradiction
    | cons x' xs xs_ih =>
      cases' ys with y ys
      · cases h₃
      dsimp [List.dlookup] at h₃; split_ifs at h₃ with h
      · rw [Option.some_inj] at h₃
        subst x'; subst val
        simp only [List.mem_cons, true_or, eq_self_iff_true]
      · cases' h₀ with _ _ h₀ h₅
        cases' h₂ with _ _ h₂ h₄
        have h₆ := Nat.succ.inj h₁
        specialize xs_ih h₅ h₃ h₄ h₆
        simp only [Ne.symm h, xs_ih, List.mem_cons]
        suffices val ∈ ys by tauto
        rw [← Option.mem_def, List.mem_dlookup_iff] at h₃
        · simp only [Prod.toSigma, List.mem_map, heq_iff_eq, Prod.exists] at h₃
          rcases h₃ with ⟨a, b, h₃, h₄, h₅⟩
          apply (List.of_mem_zip h₃).2
        simp only [List.NodupKeys, List.keys, comp_def, Prod.fst_toSigma, List.map_map]
        rwa [List.map_fst_zip _ _ (le_of_eq h₆)]


theorem List.applyId_eq_self [DecidableEq α] {xs ys : List α} (x : α) :
    x ∉ xs → List.applyId.{u} (xs.zip ys) x = x := by
  /-
    α : Type u
    inst✝ : DecidableEq α
    xs ys : List α
    x : α
    ⊢ Not (Membership.mem xs x) → Eq (Plausible.InjectiveFunction.List.applyId (xs …
  -/
  intro h
  /-
    α : Type u
    inst✝ : DecidableEq α
    xs ys : List α
    x : α
    h : Not (Membership.mem xs x)
    ⊢ Eq (Plausible.InjectiveFunction.List.applyId (xs.zip ys) x) x
  -/
  dsimp [List.applyId]
  /-
    α : Type u
    inst✝ : DecidableEq α
    xs ys : List α
    x : α
    h : Not (Membership.mem xs x)
    ⊢ Eq ((List.dlookup x (List.map Prod.toSigma (xs.zip ys))).getD x) x
  -/
  rw [List.dlookup_eq_none.2]
    /-
      α : Type u
      inst✝ : DecidableEq α
      xs ys : List α
      x : α
      h : Not (Membership.mem xs x)
      ⊢ Eq (Option.none.getD x) x
    -/
  · rfl
    /-
      🎉 no goals
    -/
  simp only [List.keys, not_exists, Prod.toSigma, exists_and_right, exists_eq_right, List.mem_map,
    Function.comp_apply, List.map_map, Prod.exists]
  /-
    α : Type u
    inst✝ : DecidableEq α
    xs ys : List α
    x : α
    h : Not (Membership.mem xs x)
    ⊢ ∀ (x_1 : α), Not (Membership.mem (xs.zip ys) { fst := x, snd := x_1 })
  -/
  intro y hy
  /-
    α : Type u
    inst✝ : DecidableEq α
    xs ys : List α
    x : α
    h : Not (Membership.mem xs x)
    y : α
    hy : Membership.mem (xs.zip ys) { fst := x, snd := y }
    ⊢ False
  -/
  exact h (List.of_mem_zip hy).1
  /-
    🎉 no goals
  -/


theorem applyId_injective [DecidableEq α] {xs ys : List α} (h₀ : List.Nodup xs) (h₁ : xs ~ ys) :
    Injective.{u + 1, u + 1} (List.applyId (xs.zip ys)) := by
  /-
    α : Type u
    inst✝ : DecidableEq α
    xs ys : List α
    h₀ : xs.Nodup
    h₁ : xs.Perm ys
    ⊢ Function.Injective (Plausible.InjectiveFunction.List.applyId (xs.zip ys))
  -/
  intro x y h
  /-
    α : Type u
    inst✝ : DecidableEq α
    xs ys : List α
    h₀ : xs.Nodup
    h₁ : xs.Perm ys
    x y : α
    h : Eq (Plausible.InjectiveFunction.List.applyId (xs.zip ys) x) (Plausible.Inj …
    ⊢ Eq x y
  -/
  by_cases hx : x ∈ xs <;> by_cases hy : y ∈ xs
    /-
      case pos
      α : Type u
      inst✝ : DecidableEq α
      xs ys : List α
      h₀ : xs.Nodup
      h₁ : xs.Perm ys
      x y : α
      h : Eq (Plausible.InjectiveFunction.List.applyId (xs.zip ys) x) (Plausible.Inj …
      hx : Membership.mem xs x
      hy : Membership.mem xs y
      ⊢ Eq x y
    -/
  · rw [List.mem_iff_getElem?] at hx hy
    /-
      case pos
      α : Type u
      inst✝ : DecidableEq α
      xs ys : List α
      h₀ : xs.Nodup
      h₁ : xs.Perm ys
      x y : α
      h : Eq (Plausible.InjectiveFunction.List.applyId (xs.zip ys) x) (Plausible.Inj …
      hx : Exists fun n => Eq (GetElem?.getElem? xs n) (Option.some x)
      hy : Exists fun n => Eq (GetElem?.getElem? xs n) (Option.some y)
      ⊢ Eq x y
    -/
    cases' hx with i hx
    /-
      case pos.intro
      α : Type u
      inst✝ : DecidableEq α
      xs ys : List α
      h₀ : xs.Nodup
      h₁ : xs.Perm ys
      x y : α
      h : Eq (Plausible.InjectiveFunction.List.applyId (xs.zip ys) x) (Plausible.Inj …
      hy : Exists fun n => Eq (GetElem?.getElem? xs n) (Option.some y)
      i : Nat
      hx : Eq (GetElem?.getElem? xs i) (Option.some x)
      ⊢ Eq x y
    -/
    cases' hy with j hy
    /-
      case pos.intro.intro
      α : Type u
      inst✝ : DecidableEq α
      xs ys : List α
      h₀ : xs.Nodup
      h₁ : xs.Perm ys
      x y : α
      h : Eq (Plausible.InjectiveFunction.List.applyId (xs.zip ys) x) (Plausible.Inj …
      i : Nat
      hx : Eq (GetElem?.getElem? xs i) (Option.some x)
      j : Nat
      hy : Eq (GetElem?.getElem? xs j) (Option.some y)
      ⊢ Eq x y
    -/
    suffices some x = some y by injection this
    /-
      case pos.intro.intro
      α : Type u
      inst✝ : DecidableEq α
      xs ys : List α
      h₀ : xs.Nodup
      h₁ : xs.Perm ys
      x y : α
      h : Eq (Plausible.InjectiveFunction.List.applyId (xs.zip ys) x) (Plausible.Inj …
      i : Nat
      hx : Eq (GetElem?.getElem? xs i) (Option.some x)
      j : Nat
      hy : Eq (GetElem?.getElem? xs j) (Option.some y)
      ⊢ Eq (Option.some x) (Option.some y)
    -/
    have h₂ := h₁.length_eq
    /-
      case pos.intro.intro
      α : Type u
      inst✝ : DecidableEq α
      xs ys : List α
      h₀ : xs.Nodup
      h₁ : xs.Perm ys
      x y : α
      h : Eq (Plausible.InjectiveFunction.List.applyId (xs.zip ys) x) (Plausible.Inj …
      i : Nat
      hx : Eq (GetElem?.getElem? xs i) (Option.some x)
      j : Nat
      hy : Eq (GetElem?.getElem? xs j) (Option.some y)
      h₂ : Eq xs.length ys.length
      ⊢ Eq (Option.some x) (Option.some y)
    -/
    rw [List.applyId_zip_eq h₀ h₂ _ _ _ hx] at h
    /-
      case pos.intro.intro
      α : Type u
      inst✝ : DecidableEq α
      xs ys : List α
      h₀ : xs.Nodup
      h₁ : xs.Perm ys
      x y : α
      i : Nat
      h : Eq (GetElem?.getElem? ys i) (Option.some (Plausible.InjectiveFunction.List …
      hx : Eq (GetElem?.getElem? xs i) (Option.some x)
      j : Nat
      hy : Eq (GetElem?.getElem? xs j) (Option.some y)
      h₂ : Eq xs.length ys.length
      ⊢ Eq (Option.some x) (Option.some y)
    -/
    rw [← hx, ← hy]; congr
    /-
      case pos.intro.intro.e_a
      α : Type u
      inst✝ : DecidableEq α
      xs ys : List α
      h₀ : xs.Nodup
      h₁ : xs.Perm ys
      x y : α
      i : Nat
      h : Eq (GetElem?.getElem? ys i) (Option.some (Plausible.InjectiveFunction.List …
      hx : Eq (GetElem?.getElem? xs i) (Option.some x)
      j : Nat
      hy : Eq (GetElem?.getElem? xs j) (Option.some y)
      h₂ : Eq xs.length ys.length
      ⊢ Eq i j
    -/
    apply List.getElem?_inj _ (h₁.nodup_iff.1 h₀)
      /-
        case pos.intro.intro.e_a
        α : Type u
        inst✝ : DecidableEq α
        xs ys : List α
        h₀ : xs.Nodup
        h₁ : xs.Perm ys
        x y : α
        i : Nat
        h : Eq (GetElem?.getElem? ys i) (Option.some (Plausible.InjectiveFunction.List …
        hx : Eq (GetElem?.getElem? xs i) (Option.some x)
        j : Nat
        hy : Eq (GetElem?.getElem? xs j) (Option.some y)
        h₂ : Eq xs.length ys.length
        ⊢ Eq (GetElem?.getElem? ys i) (GetElem?.getElem? ys j)
      -/
    · symm; rw [h]
      /-
        case pos.intro.intro.e_a
        α : Type u
        inst✝ : DecidableEq α
        xs ys : List α
        h₀ : xs.Nodup
        h₁ : xs.Perm ys
        x y : α
        i : Nat
        h : Eq (GetElem?.getElem? ys i) (Option.some (Plausible.InjectiveFunction.List …
        hx : Eq (GetElem?.getElem? xs i) (Option.some x)
        j : Nat
        hy : Eq (GetElem?.getElem? xs j) (Option.some y)
        h₂ : Eq xs.length ys.length
        ⊢ Eq (GetElem?.getElem? ys j) (Option.some (Plausible.InjectiveFunction.List.a …
      -/
                                     /-
                                       🎉 no goals
                                     -/
                                     /-
                                       🎉 no goals
                                     -/
      rw [← List.applyId_zip_eq] <;> assumption
                                     /-
                                       🎉 no goals
                                     -/
      /-
        α : Type u
        inst✝ : DecidableEq α
        xs ys : List α
        h₀ : xs.Nodup
        h₁ : xs.Perm ys
        x y : α
        i : Nat
        h : Eq (GetElem?.getElem? ys i) (Option.some (Plausible.InjectiveFunction.List …
        hx : Eq (GetElem?.getElem? xs i) (Option.some x)
        j : Nat
        hy : Eq (GetElem?.getElem? xs j) (Option.some y)
        h₂ : Eq xs.length ys.length
        ⊢ LT.lt i ys.length
      -/
    · rw [← h₁.length_eq]
      /-
        α : Type u
        inst✝ : DecidableEq α
        xs ys : List α
        h₀ : xs.Nodup
        h₁ : xs.Perm ys
        x y : α
        i : Nat
        h : Eq (GetElem?.getElem? ys i) (Option.some (Plausible.InjectiveFunction.List …
        hx : Eq (GetElem?.getElem? xs i) (Option.some x)
        j : Nat
        hy : Eq (GetElem?.getElem? xs j) (Option.some y)
        h₂ : Eq xs.length ys.length
        ⊢ LT.lt i xs.length
      -/
      rw [List.getElem?_eq_some_iff] at hx
      /-
        α : Type u
        inst✝ : DecidableEq α
        xs ys : List α
        h₀ : xs.Nodup
        h₁ : xs.Perm ys
        x y : α
        i : Nat
        h : Eq (GetElem?.getElem? ys i) (Option.some (Plausible.InjectiveFunction.List …
        hx : Exists fun h => Eq (GetElem.getElem xs i h) x
        j : Nat
        hy : Eq (GetElem?.getElem? xs j) (Option.some y)
        h₂ : Eq xs.length ys.length
        ⊢ LT.lt i xs.length
      -/
      cases' hx with hx hx'
      /-
        case intro
        α : Type u
        inst✝ : DecidableEq α
        xs ys : List α
        h₀ : xs.Nodup
        h₁ : xs.Perm ys
        x y : α
        i : Nat
        h : Eq (GetElem?.getElem? ys i) (Option.some (Plausible.InjectiveFunction.List …
        j : Nat
        hy : Eq (GetElem?.getElem? xs j) (Option.some y)
        h₂ : Eq xs.length ys.length
        hx : LT.lt i xs.length
        hx' : Eq (GetElem.getElem xs i hx) x
        ⊢ LT.lt i xs.length
      -/
      exact hx
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u
      inst✝ : DecidableEq α
      xs ys : List α
      h₀ : xs.Nodup
      h₁ : xs.Perm ys
      x y : α
      h : Eq (Plausible.InjectiveFunction.List.applyId (xs.zip ys) x) (Plausible.Inj …
      hx : Membership.mem xs x
      hy : Not (Membership.mem xs y)
      ⊢ Eq x y
    -/
  · rw [← applyId_mem_iff h₀ h₁] at hx hy
    /-
      case neg
      α : Type u
      inst✝ : DecidableEq α
      xs ys : List α
      h₀ : xs.Nodup
      h₁ : xs.Perm ys
      x y : α
      h : Eq (Plausible.InjectiveFunction.List.applyId (xs.zip ys) x) (Plausible.Inj …
      hx : Membership.mem ys (Plausible.InjectiveFunction.List.applyId (xs.zip ys) x)
      hy : Not (Membership.mem ys (Plausible.InjectiveFunction.List.applyId (xs.zip  …
      ⊢ Eq x y
    -/
    rw [h] at hx
    /-
      case neg
      α : Type u
      inst✝ : DecidableEq α
      xs ys : List α
      h₀ : xs.Nodup
      h₁ : xs.Perm ys
      x y : α
      h : Eq (Plausible.InjectiveFunction.List.applyId (xs.zip ys) x) (Plausible.Inj …
      hx : Membership.mem ys (Plausible.InjectiveFunction.List.applyId (xs.zip ys) y)
      hy : Not (Membership.mem ys (Plausible.InjectiveFunction.List.applyId (xs.zip  …
      ⊢ Eq x y
    -/
    contradiction
    /-
      🎉 no goals
    -/
    /-
      case pos
      α : Type u
      inst✝ : DecidableEq α
      xs ys : List α
      h₀ : xs.Nodup
      h₁ : xs.Perm ys
      x y : α
      h : Eq (Plausible.InjectiveFunction.List.applyId (xs.zip ys) x) (Plausible.Inj …
      hx : Not (Membership.mem xs x)
      hy : Membership.mem xs y
      ⊢ Eq x y
    -/
  · rw [← applyId_mem_iff h₀ h₁] at hx hy
    /-
      case pos
      α : Type u
      inst✝ : DecidableEq α
      xs ys : List α
      h₀ : xs.Nodup
      h₁ : xs.Perm ys
      x y : α
      h : Eq (Plausible.InjectiveFunction.List.applyId (xs.zip ys) x) (Plausible.Inj …
      hx : Not (Membership.mem ys (Plausible.InjectiveFunction.List.applyId (xs.zip  …
      hy : Membership.mem ys (Plausible.InjectiveFunction.List.applyId (xs.zip ys) y)
      ⊢ Eq x y
    -/
    rw [h] at hx
    /-
      case pos
      α : Type u
      inst✝ : DecidableEq α
      xs ys : List α
      h₀ : xs.Nodup
      h₁ : xs.Perm ys
      x y : α
      h : Eq (Plausible.InjectiveFunction.List.applyId (xs.zip ys) x) (Plausible.Inj …
      hx : Not (Membership.mem ys (Plausible.InjectiveFunction.List.applyId (xs.zip  …
      hy : Membership.mem ys (Plausible.InjectiveFunction.List.applyId (xs.zip ys) y)
      ⊢ Eq x y
    -/
    contradiction
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u
      inst✝ : DecidableEq α
      xs ys : List α
      h₀ : xs.Nodup
      h₁ : xs.Perm ys
      x y : α
      h : Eq (Plausible.InjectiveFunction.List.applyId (xs.zip ys) x) (Plausible.Inj …
      hx : Not (Membership.mem xs x)
      hy : Not (Membership.mem xs y)
      ⊢ Eq x y
    -/
                                                              /-
                                                                🎉 no goals
                                                              -/
  · rwa [List.applyId_eq_self, List.applyId_eq_self] at h <;> assumption
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- Remove a slice of length `m` at index `n` in a list and a permutation, maintaining the property
that it is a permutation.
-/
def Perm.slice [DecidableEq α] (n m : ℕ) :
    (Σ' xs ys : List α, xs ~ ys ∧ ys.Nodup) → Σ' xs ys : List α, xs ~ ys ∧ ys.Nodup
  | ⟨xs, ys, h, h'⟩ =>
    let xs' := List.dropSlice n m xs
    have h₀ : xs' ~ ys.inter xs' := List.Perm.dropSlice_inter _ _ h h'
    ⟨xs', ys.inter xs', h₀, h'.inter _⟩


/-- A list, in decreasing order, of sizes that should be
sliced off a list of length `n`
-/
def sliceSizes : ℕ → MLList Id ℕ+
  | n =>
    if h : 0 < n then
                                                /-
                                                  α : Type u
                                                  β : Type v
                                                  x✝ : Nat
                                                  n : Nat := x✝
                                                  h : LT.lt 0 n
                                                  ⊢ LT.lt 1 2
                                                -/
      have : n / 2 < n := Nat.div_lt_self h (by decide : 1 < 2)
                                                /-
                                                  🎉 no goals
                                                -/
      .cons ⟨_, h⟩ (sliceSizes <| n / 2)
    else .nil


/-- Shrink a permutation of a list, slicing a segment in the middle.

The sizes of the slice being removed start at `n` (with `n` the length
of the list) and then `n / 2`, then `n / 4`, etc down to 1. The slices
will be taken at index `0`, `n / k`, `2n / k`, `3n / k`, etc.
-/
protected def shrinkPerm {α : Type} [DecidableEq α] :
    (Σ' xs ys : List α, xs ~ ys ∧ ys.Nodup) → List (Σ' xs ys : List α, xs ~ ys ∧ ys.Nodup)
  | xs => do
    let k := xs.1.length
    let n ← (sliceSizes k).force
    let i ← List.finRange <| k / n
    pure <| Perm.slice (i * n) n xs


-- Porting note: removed, there is no `sizeof` in the new `Sampleable`
-- instance [SizeOf α] : SizeOf (InjectiveFunction α) :=
--   ⟨fun ⟨xs, _, _⟩ => SizeOf.sizeOf (xs.map Sigma.fst)⟩


/-- Shrink an injective function slicing a segment in the middle of the domain and removing
the corresponding elements in the codomain, hence maintaining the property that
one is a permutation of the other.
-/
protected def shrink {α : Type} [DecidableEq α] :
    InjectiveFunction α → List (InjectiveFunction α)
  | ⟨_, h₀, h₁⟩ => do
    let ⟨xs', ys', h₀, h₁⟩ ← InjectiveFunction.shrinkPerm ⟨_, _, h₀, h₁⟩
    have h₃ : xs'.length ≤ ys'.length := le_of_eq (List.Perm.length_eq h₀)
    have h₄ : ys'.length ≤ xs'.length := le_of_eq (List.Perm.length_eq h₀.symm)
    pure
      ⟨(List.zip xs' ys').map Prod.toSigma,
        by simp only [comp_def, List.map_fst_zip, List.map_snd_zip, *, Prod.fst_toSigma,
          Prod.snd_toSigma, List.map_map],
           /-
             α✝ : Type u
             β : Type v
             α : Type
             inst✝ : DecidableEq α
             xs✝ : List (Sigma fun x => α)
             h₀✝ : (List.map Sigma.fst xs✝).Perm (List.map Sigma.snd xs✝)
             h₁✝ : (List.map Sigma.snd xs✝).Nodup
             xs' ys' : List α
             h₀ : xs'.Perm ys'
             h₁ : ys'.Nodup
             h₃ : LE.le xs'.length ys'.length
             h₄ : LE.le ys'.length xs'.length
             ⊢ (List.map Sigma.snd (List.map Prod.toSigma (xs'.zip ys'))).Nodup
           -/
        by simp only [comp_def, List.map_snd_zip, *, Prod.snd_toSigma, List.map_map]⟩
           /-
             🎉 no goals
           -/


/-- Create an injective function from one list and a permutation of that list. -/
protected def mk (xs ys : List α) (h : xs ~ ys) (h' : ys.Nodup) : InjectiveFunction α :=
  have h₀ : xs.length ≤ ys.length := le_of_eq h.length_eq
  have h₁ : ys.length ≤ xs.length := le_of_eq h.length_eq.symm
  InjectiveFunction.mapToSelf (List.toFinmap' (xs.zip ys))
    (by
      simp only [List.toFinmap', comp_def, List.map_fst_zip, List.map_snd_zip, *, Prod.fst_toSigma,
        Prod.snd_toSigma, List.map_map])
        /-
          α : Type u
          β : Type v
          xs ys : List α
          h : xs.Perm ys
          h' : ys.Nodup
          h₀ : LE.le xs.length ys.length
          h₁ : LE.le ys.length xs.length
          ⊢ (List.map Sigma.snd (Plausible.TotalFunction.List.toFinmap' (xs.zip ys))).No …
        -/
    (by simp only [List.toFinmap', comp_def, List.map_snd_zip, *, Prod.snd_toSigma, List.map_map])
        /-
          🎉 no goals
        -/


protected theorem injective [DecidableEq α] (f : InjectiveFunction α) : Injective (apply f) := by
  /-
    α : Type u
    inst✝ : DecidableEq α
    f : Plausible.InjectiveFunction α
    ⊢ Function.Injective f.apply
  -/
  cases' f with xs hperm hnodup
  /-
    case mapToSelf
    α : Type u
    inst✝ : DecidableEq α
    xs : List (Sigma fun x => α)
    hperm : (List.map Sigma.fst xs).Perm (List.map Sigma.snd xs)
    hnodup : (List.map Sigma.snd xs).Nodup
    ⊢ Function.Injective (Plausible.InjectiveFunction.mapToSelf xs hperm hnodup).a …
  -/
  generalize h₀ : List.map Sigma.fst xs = xs₀
  /-
    case mapToSelf
    α : Type u
    inst✝ : DecidableEq α
    xs : List (Sigma fun x => α)
    hperm : (List.map Sigma.fst xs).Perm (List.map Sigma.snd xs)
    hnodup : (List.map Sigma.snd xs).Nodup
    xs₀ : List α
    h₀ : Eq (List.map Sigma.fst xs) xs₀
    ⊢ Function.Injective (Plausible.InjectiveFunction.mapToSelf xs hperm hnodup).a …
  -/
  generalize h₁ : xs.map (@id ((Σ _ : α, α) → α) <| @Sigma.snd α fun _ : α => α) = xs₁
  /-
    case mapToSelf
    α : Type u
    inst✝ : DecidableEq α
    xs : List (Sigma fun x => α)
    hperm : (List.map Sigma.fst xs).Perm (List.map Sigma.snd xs)
    hnodup : (List.map Sigma.snd xs).Nodup
    xs₀ : List α
    h₀ : Eq (List.map Sigma.fst xs) xs₀
    xs₁ : List α
    h₁ : Eq (List.map (id Sigma.snd) xs) xs₁
    ⊢ Function.Injective (Plausible.InjectiveFunction.mapToSelf xs hperm hnodup).a …
  -/
  dsimp [id] at h₁
  have hxs : xs = TotalFunction.List.toFinmap' (xs₀.zip xs₁) := by
    rw [← h₀, ← h₁, List.toFinmap']; clear h₀ h₁ xs₀ xs₁ hperm hnodup
    induction xs with
    | nil => simp only [List.zip_nil_right, List.map_nil]
    | cons xs_hd xs_tl xs_ih =>
      simp only [Prod.toSigma, eq_self_iff_true, Sigma.eta, List.zip_cons_cons,
        List.map, List.cons_inj_right]
      exact xs_ih
  /-
    case mapToSelf
    α : Type u
    inst✝ : DecidableEq α
    xs : List (Sigma fun x => α)
    hperm : (List.map Sigma.fst xs).Perm (List.map Sigma.snd xs)
    hnodup : (List.map Sigma.snd xs).Nodup
    xs₀ : List α
    h₀ : Eq (List.map Sigma.fst xs) xs₀
    xs₁ : List α
    h₁ : Eq (List.map Sigma.snd xs) xs₁
    hxs : Eq xs (Plausible.TotalFunction.List.toFinmap' (xs₀.zip xs₁))
    ⊢ Function.Injective (Plausible.InjectiveFunction.mapToSelf xs hperm hnodup).a …
  -/
  revert hperm hnodup
  /-
    case mapToSelf
    α : Type u
    inst✝ : DecidableEq α
    xs : List (Sigma fun x => α)
    xs₀ : List α
    h₀ : Eq (List.map Sigma.fst xs) xs₀
    xs₁ : List α
    h₁ : Eq (List.map Sigma.snd xs) xs₁
    hxs : Eq xs (Plausible.TotalFunction.List.toFinmap' (xs₀.zip xs₁))
    ⊢ ∀ (hperm : (List.map Sigma.fst xs).Perm (List.map Sigma.snd xs)) (hnodup : ( …
  -/
  rw [hxs]; intros hperm hnodup
  /-
    case mapToSelf
    α : Type u
    inst✝ : DecidableEq α
    xs : List (Sigma fun x => α)
    xs₀ : List α
    h₀ : Eq (List.map Sigma.fst xs) xs₀
    xs₁ : List α
    h₁ : Eq (List.map Sigma.snd xs) xs₁
    hxs : Eq xs (Plausible.TotalFunction.List.toFinmap' (xs₀.zip xs₁))
    hperm : (List.map Sigma.fst (Plausible.TotalFunction.List.toFinmap' (xs₀.zip x …
    hnodup : (List.map Sigma.snd (Plausible.TotalFunction.List.toFinmap' (xs₀.zip  …
    ⊢ Function.Injective (Plausible.InjectiveFunction.mapToSelf (Plausible.TotalFu …
  -/
  apply InjectiveFunction.applyId_injective
    /-
      case mapToSelf.h₀
      α : Type u
      inst✝ : DecidableEq α
      xs : List (Sigma fun x => α)
      xs₀ : List α
      h₀ : Eq (List.map Sigma.fst xs) xs₀
      xs₁ : List α
      h₁ : Eq (List.map Sigma.snd xs) xs₁
      hxs : Eq xs (Plausible.TotalFunction.List.toFinmap' (xs₀.zip xs₁))
      hperm : (List.map Sigma.fst (Plausible.TotalFunction.List.toFinmap' (xs₀.zip x …
      hnodup : (List.map Sigma.snd (Plausible.TotalFunction.List.toFinmap' (xs₀.zip  …
      ⊢ xs₀.Nodup
    -/
  · rwa [← h₀, hxs, hperm.nodup_iff]
    /-
      🎉 no goals
    -/
    /-
      case mapToSelf.h₁
      α : Type u
      inst✝ : DecidableEq α
      xs : List (Sigma fun x => α)
      xs₀ : List α
      h₀ : Eq (List.map Sigma.fst xs) xs₀
      xs₁ : List α
      h₁ : Eq (List.map Sigma.snd xs) xs₁
      hxs : Eq xs (Plausible.TotalFunction.List.toFinmap' (xs₀.zip xs₁))
      hperm : (List.map Sigma.fst (Plausible.TotalFunction.List.toFinmap' (xs₀.zip x …
      hnodup : (List.map Sigma.snd (Plausible.TotalFunction.List.toFinmap' (xs₀.zip  …
      ⊢ xs₀.Perm xs₁
    -/
  · rwa [← hxs, h₀, h₁] at hperm
    /-
      🎉 no goals
    -/


instance PiInjective.sampleableExt : SampleableExt { f : ℤ → ℤ // Function.Injective f } where
  proxy := InjectiveFunction ℤ
  interp f := ⟨apply f, f.injective⟩
  sample := do
    let ⟨sz⟩ ← Gen.up Gen.getSize
    let xs' := Int.range (-(2 * sz + 2)) (2 * sz + 2)
    let ys ← Gen.permutationOf xs'
    have Hinj : Injective fun r : ℕ => -(2 * sz + 2 : ℤ) + ↑r := fun _x _y h =>
        Int.ofNat.inj (add_right_injective _ h)
    let r : InjectiveFunction ℤ :=
      InjectiveFunction.mk.{0} xs' ys.1 ys.2 (ys.2.nodup_iff.1 <| (List.nodup_range _).map Hinj)
    pure r
  shrink := {shrink := @InjectiveFunction.shrink ℤ _ }


instance Injective.testable (f : α → β)
    [I : Testable (NamedBinder "x" <|
      ∀ x : α, NamedBinder "y" <| ∀ y : α, NamedBinder "H" <| f x = f y → x = y)] :
    Testable (Injective f) :=
  I


instance Monotone.testable [Preorder α] [Preorder β] (f : α → β)
    [I : Testable (NamedBinder "x" <|
      ∀ x : α, NamedBinder "y" <| ∀ y : α, NamedBinder "H" <| x ≤ y → f x ≤ f y)] :
    Testable (Monotone f) :=
  I


instance Antitone.testable [Preorder α] [Preorder β] (f : α → β)
    [I : Testable (NamedBinder "x" <|
      ∀ x : α, NamedBinder "y" <| ∀ y : α, NamedBinder "H" <| x ≤ y → f y ≤ f x)] :
    Testable (Antitone f) :=
  I


