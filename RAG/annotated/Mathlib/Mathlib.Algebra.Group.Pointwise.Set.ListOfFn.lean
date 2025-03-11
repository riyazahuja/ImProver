@[to_additive]
theorem mem_prod_list_ofFn {a : α} {s : Fin n → Set α} :
    a ∈ (List.ofFn s).prod ↔ ∃ f : ∀ i : Fin n, s i, (List.ofFn fun i ↦ (f i : α)).prod = a := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    n : Nat
    a : α
    s : Fin n → Set α
    ⊢ Iff (Membership.mem (List.ofFn s).prod a) (Exists fun f => Eq (List.ofFn fun …
  -/
  induction' n with n ih generalizing a
    /-
      case zero
      α : Type u_1
      inst✝ : Monoid α
      n : Nat
      a : α
      s : Fin 0 → Set α
      ⊢ Iff (Membership.mem (List.ofFn s).prod a) (Exists fun f => Eq (List.ofFn fun …
    -/
  · simp_rw [List.ofFn_zero, List.prod_nil, Fin.exists_fin_zero_pi, eq_comm, Set.mem_one]
    /-
      🎉 no goals
    -/
  · simp_rw [List.ofFn_succ, List.prod_cons, Fin.exists_fin_succ_pi, Fin.cons_zero, Fin.cons_succ,
      mem_mul, @ih, exists_exists_eq_and, SetCoe.exists, exists_prop]


@[to_additive]
theorem mem_list_prod {l : List (Set α)} {a : α} :
    a ∈ l.prod ↔
      ∃ l' : List (Σs : Set α, ↥s),
        List.prod (l'.map fun x ↦ (Sigma.snd x : α)) = a ∧ l'.map Sigma.fst = l := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    l : List (Set α)
    a : α
    ⊢ Iff (Membership.mem l.prod a) (Exists fun l' => And (Eq (List.map (fun x =>  …
  -/
  induction' l using List.ofFnRec with n f
  simp only [mem_prod_list_ofFn, List.exists_iff_exists_tuple, List.map_ofFn, Function.comp,
    List.ofFn_inj', Sigma.mk.inj_iff, and_left_comm, exists_and_left, exists_eq_left, heq_eq_eq]
  /-
    case h
    α : Type u_1
    inst✝ : Monoid α
    a : α
    n : Nat
    f : Fin n → Set α
    ⊢ Iff (Exists fun f_1 => Eq (List.ofFn fun i => ↑(f_1 i)).prod a) (Exists fun  …
  -/
  constructor
    /-
      case h.mp
      α : Type u_1
      inst✝ : Monoid α
      a : α
      n : Nat
      f : Fin n → Set α
      ⊢ (Exists fun f_1 => Eq (List.ofFn fun i => ↑(f_1 i)).prod a) → Exists fun x = …
    -/
  · rintro ⟨fi, rfl⟩
    /-
      case h.mp.intro
      α : Type u_1
      inst✝ : Monoid α
      n : Nat
      f : Fin n → Set α
      fi : (i : Fin n) → ↑(f i)
      ⊢ Exists fun x => And (Eq (List.ofFn (Function.comp (fun x => ↑x.snd) x)).prod …
    -/
    exact ⟨fun i ↦ ⟨_, fi i⟩, rfl, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u_1
      inst✝ : Monoid α
      a : α
      n : Nat
      f : Fin n → Set α
      ⊢ (Exists fun x => And (Eq (List.ofFn (Function.comp (fun x => ↑x.snd) x)).pro …
    -/
  · rintro ⟨fi, rfl, rfl⟩
    /-
      case h.mpr.intro.intro
      α : Type u_1
      inst✝ : Monoid α
      n : Nat
      fi : Fin n → Sigma fun s => ↑s
      ⊢ Exists fun f => Eq (List.ofFn fun i => ↑(f i)).prod (List.ofFn (Function.com …
    -/
    exact ⟨fun i ↦ _, rfl⟩
    /-
      🎉 no goals
    -/


@[to_additive]
theorem mem_pow {a : α} {n : ℕ} :
    a ∈ s ^ n ↔ ∃ f : Fin n → s, (List.ofFn fun i ↦ (f i : α)).prod = a := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    s : Set α
    a : α
    n : Nat
    ⊢ Iff (Membership.mem (HPow.hPow s n) a) (Exists fun f => Eq (List.ofFn fun i  …
  -/
  rw [← mem_prod_list_ofFn, List.ofFn_const, List.prod_replicate]
  /-
    🎉 no goals
  -/


