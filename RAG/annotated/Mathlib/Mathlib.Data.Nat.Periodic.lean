theorem periodic_gcd (a : ℕ) : Periodic (gcd a) a := by
  /-
    a : Nat
    ⊢ Function.Periodic a.gcd a
  -/
  simp only [forall_const, gcd_add_self_right, eq_self_iff_true, Periodic]
  /-
    🎉 no goals
  -/


theorem periodic_coprime (a : ℕ) : Periodic (Coprime a) a := by
  /-
    a : Nat
    ⊢ Function.Periodic a.Coprime a
  -/
  simp only [coprime_add_self_right, forall_const, eq_iff_iff, Periodic]
  /-
    🎉 no goals
  -/


theorem periodic_mod (a : ℕ) : Periodic (fun n => n % a) a := by
  /-
    a : Nat
    ⊢ Function.Periodic (fun n => HMod.hMod n a) a
  -/
  simp only [forall_const, eq_self_iff_true, add_mod_right, Periodic]
  /-
    🎉 no goals
  -/


theorem _root_.Function.Periodic.map_mod_nat {α : Type*} {f : ℕ → α} {a : ℕ} (hf : Periodic f a) :
    ∀ n, f (n % a) = f n := fun n => by
  /-
    α : Type u_1
    f : Nat → α
    a : Nat
    hf : Function.Periodic f a
    n : Nat
    ⊢ Eq (f (HMod.hMod n a)) (f n)
  -/
  conv_rhs => rw [← Nat.mod_add_div n a, mul_comm, ← Nat.nsmul_eq_mul, hf.nsmul]
  /-
    🎉 no goals
  -/


/-- An interval of length `a` filtered over a periodic predicate of period `a` has cardinality
equal to the number naturals below `a` for which `p a` is true. -/
theorem filter_multiset_Ico_card_eq_of_periodic (n a : ℕ) (p : ℕ → Prop) [DecidablePred p]
    (pp : Periodic p a) : card (filter p (Ico n (n + a))) = a.count p := by
  rw [count_eq_card_filter_range, Finset.card, Finset.filter_val, Finset.range_val, ←
    multiset_Ico_map_mod n, ← map_count_True_eq_filter_card, ← map_count_True_eq_filter_card,
    map_map]
  /-
    n a : Nat
    p : Nat → Prop
    inst✝ : DecidablePred p
    pp : Function.Periodic p a
    ⊢ Eq (Multiset.count True (Multiset.map p (Multiset.Ico n (HAdd.hAdd n a)))) ( …
  -/
  congr; funext n
  /-
    case e_a.e_f.h
    n✝ a : Nat
    p : Nat → Prop
    inst✝ : DecidablePred p
    pp : Function.Periodic p a
    n : Nat
    ⊢ Eq (p n) (Function.comp (fun x => p x) (fun x => HMod.hMod x a) n)
  -/
  exact (Function.Periodic.map_mod_nat pp n).symm
  /-
    🎉 no goals
  -/


/-- An interval of length `a` filtered over a periodic predicate of period `a` has cardinality
equal to the number naturals below `a` for which `p a` is true. -/
theorem filter_Ico_card_eq_of_periodic (n a : ℕ) (p : ℕ → Prop) [DecidablePred p]
    (pp : Periodic p a) : ((Ico n (n + a)).filter p).card = a.count p :=
  filter_multiset_Ico_card_eq_of_periodic n a p pp


