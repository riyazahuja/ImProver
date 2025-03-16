/-- An ordinal `o` is said to be principal or indecomposable under an operation when the set of
ordinals less than it is closed under that operation. In standard mathematical usage, this term is
almost exclusively used for additive and multiplicative principal ordinals.

For simplicity, we break usual convention and regard `0` as principal. -/
def Principal (op : Ordinal → Ordinal → Ordinal) (o : Ordinal) : Prop :=
  ∀ ⦃a b⦄, a < o → b < o → op a b < o


theorem principal_swap_iff : Principal (Function.swap op) o ↔ Principal op o := by
  /-
    o : Ordinal.{u}
    op : Ordinal.{u} → Ordinal.{u} → Ordinal.{u}
    ⊢ Iff (Ordinal.Principal (Function.swap op) o) (Ordinal.Principal op o)
  -/
                  /-
                    🎉 no goals
                  -/
  constructor <;> exact fun h a b ha hb => h hb ha
                  /-
                    🎉 no goals
                  -/


@[deprecated principal_swap_iff (since := "2024-08-18")]
theorem principal_iff_principal_swap : Principal op o ↔ Principal (Function.swap op) o :=
  principal_swap_iff


theorem not_principal_iff : ¬ Principal op o ↔ ∃ a < o, ∃ b < o, o ≤ op a b := by
  /-
    o : Ordinal.{u}
    op : Ordinal.{u} → Ordinal.{u} → Ordinal.{u}
    ⊢ Iff (Not (Ordinal.Principal op o)) (Exists fun a => And (LT.lt a o) (Exists  …
  -/
  simp [Principal]
  /-
    🎉 no goals
  -/


theorem principal_iff_of_monotone
    (h₁ : ∀ a, Monotone (op a)) (h₂ : ∀ a, Monotone (Function.swap op a)) :
    Principal op o ↔ ∀ a < o, op a a < o := by
  /-
    o : Ordinal.{u}
    op : Ordinal.{u} → Ordinal.{u} → Ordinal.{u}
    h₁ : ∀ (a : Ordinal.{u}), Monotone (op a)
    h₂ : ∀ (a : Ordinal.{u}), Monotone (Function.swap op a)
    ⊢ Iff (Ordinal.Principal op o) (∀ (a : Ordinal.{u}), LT.lt a o → LT.lt (op a a …
  -/
  use fun h a ha => h ha ha
  /-
    case mpr
    o : Ordinal.{u}
    op : Ordinal.{u} → Ordinal.{u} → Ordinal.{u}
    h₁ : ∀ (a : Ordinal.{u}), Monotone (op a)
    h₂ : ∀ (a : Ordinal.{u}), Monotone (Function.swap op a)
    ⊢ (∀ (a : Ordinal.{u}), LT.lt a o → LT.lt (op a a) o) → Ordinal.Principal op o
  -/
  intro H a b ha hb
  /-
    case mpr
    o : Ordinal.{u}
    op : Ordinal.{u} → Ordinal.{u} → Ordinal.{u}
    h₁ : ∀ (a : Ordinal.{u}), Monotone (op a)
    h₂ : ∀ (a : Ordinal.{u}), Monotone (Function.swap op a)
    H : ∀ (a : Ordinal.{u}), LT.lt a o → LT.lt (op a a) o
    a b : Ordinal.{u}
    ha : LT.lt a o
    hb : LT.lt b o
    ⊢ LT.lt (op a b) o
  -/
  obtain hab | hba := le_or_lt a b
    /-
      case mpr.inl
      o : Ordinal.{u}
      op : Ordinal.{u} → Ordinal.{u} → Ordinal.{u}
      h₁ : ∀ (a : Ordinal.{u}), Monotone (op a)
      h₂ : ∀ (a : Ordinal.{u}), Monotone (Function.swap op a)
      H : ∀ (a : Ordinal.{u}), LT.lt a o → LT.lt (op a a) o
      a b : Ordinal.{u}
      ha : LT.lt a o
      hb : LT.lt b o
      hab : LE.le a b
      ⊢ LT.lt (op a b) o
    -/
  · exact (h₂ b hab).trans_lt <| H b hb
    /-
      🎉 no goals
    -/
    /-
      case mpr.inr
      o : Ordinal.{u}
      op : Ordinal.{u} → Ordinal.{u} → Ordinal.{u}
      h₁ : ∀ (a : Ordinal.{u}), Monotone (op a)
      h₂ : ∀ (a : Ordinal.{u}), Monotone (Function.swap op a)
      H : ∀ (a : Ordinal.{u}), LT.lt a o → LT.lt (op a a) o
      a b : Ordinal.{u}
      ha : LT.lt a o
      hb : LT.lt b o
      hba : LT.lt b a
      ⊢ LT.lt (op a b) o
    -/
  · exact (h₁ a hba.le).trans_lt <| H a ha
    /-
      🎉 no goals
    -/


theorem not_principal_iff_of_monotone
    (h₁ : ∀ a, Monotone (op a)) (h₂ : ∀ a, Monotone (Function.swap op a)) :
    ¬ Principal op o ↔ ∃ a < o, o ≤ op a a := by
  /-
    o : Ordinal.{u}
    op : Ordinal.{u} → Ordinal.{u} → Ordinal.{u}
    h₁ : ∀ (a : Ordinal.{u}), Monotone (op a)
    h₂ : ∀ (a : Ordinal.{u}), Monotone (Function.swap op a)
    ⊢ Iff (Not (Ordinal.Principal op o)) (Exists fun a => And (LT.lt a o) (LE.le o …
  -/
  simp [principal_iff_of_monotone h₁ h₂]
  /-
    🎉 no goals
  -/


theorem principal_zero : Principal op 0 := fun a _ h =>
  (Ordinal.not_lt_zero a h).elim


@[simp]
theorem principal_one_iff : Principal op 1 ↔ op 0 0 = 0 := by
  /-
    op : Ordinal.{u_1} → Ordinal.{u_1} → Ordinal.{u_1}
    ⊢ Iff (Ordinal.Principal op 1) (Eq (op 0 0) 0)
  -/
  refine ⟨fun h => ?_, fun h a b ha hb => ?_⟩
    /-
      case refine_1
      op : Ordinal.{u_1} → Ordinal.{u_1} → Ordinal.{u_1}
      h : Ordinal.Principal op 1
      ⊢ Eq (op 0 0) 0
    -/
  · rw [← lt_one_iff_zero]
    /-
      case refine_1
      op : Ordinal.{u_1} → Ordinal.{u_1} → Ordinal.{u_1}
      h : Ordinal.Principal op 1
      ⊢ LT.lt (op 0 0) 1
    -/
    exact h zero_lt_one zero_lt_one
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      op : Ordinal.{u_1} → Ordinal.{u_1} → Ordinal.{u_1}
      h : Eq (op 0 0) 0
      a b : Ordinal.{u_1}
      ha : LT.lt a 1
      hb : LT.lt b 1
      ⊢ LT.lt (op a b) 1
    -/
  · rwa [lt_one_iff_zero, ha, hb] at *
    /-
      🎉 no goals
    -/


theorem Principal.iterate_lt (hao : a < o) (ho : Principal op o) (n : ℕ) : (op a)^[n] a < o := by
  /-
    a o : Ordinal.{u}
    op : Ordinal.{u} → Ordinal.{u} → Ordinal.{u}
    hao : LT.lt a o
    ho : Ordinal.Principal op o
    n : Nat
    ⊢ LT.lt (Nat.iterate (op a) n a) o
  -/
  induction' n with n hn
    /-
      case zero
      a o : Ordinal.{u}
      op : Ordinal.{u} → Ordinal.{u} → Ordinal.{u}
      hao : LT.lt a o
      ho : Ordinal.Principal op o
      ⊢ LT.lt (Nat.iterate (op a) 0 a) o
    -/
  · rwa [Function.iterate_zero]
    /-
      🎉 no goals
    -/
    /-
      case succ
      a o : Ordinal.{u}
      op : Ordinal.{u} → Ordinal.{u} → Ordinal.{u}
      hao : LT.lt a o
      ho : Ordinal.Principal op o
      n : Nat
      hn : LT.lt (Nat.iterate (op a) n a) o
      ⊢ LT.lt (Nat.iterate (op a) (HAdd.hAdd n 1) a) o
    -/
  · rw [Function.iterate_succ']
    /-
      case succ
      a o : Ordinal.{u}
      op : Ordinal.{u} → Ordinal.{u} → Ordinal.{u}
      hao : LT.lt a o
      ho : Ordinal.Principal op o
      n : Nat
      hn : LT.lt (Nat.iterate (op a) n a) o
      ⊢ LT.lt (Function.comp (op a) (Nat.iterate (op a) n) a) o
    -/
    exact ho hao hn
    /-
      🎉 no goals
    -/


theorem op_eq_self_of_principal (hao : a < o) (H : IsNormal (op a))
    (ho : Principal op o) (ho' : IsLimit o) : op a o = o := by
  /-
    a o : Ordinal.{u}
    op : Ordinal.{u} → Ordinal.{u} → Ordinal.{u}
    hao : LT.lt a o
    H : Ordinal.IsNormal (op a)
    ho : Ordinal.Principal op o
    ho' : o.IsLimit
    ⊢ Eq (op a o) o
  -/
  apply H.le_apply.antisymm'
  /-
    a o : Ordinal.{u}
    op : Ordinal.{u} → Ordinal.{u} → Ordinal.{u}
    hao : LT.lt a o
    H : Ordinal.IsNormal (op a)
    ho : Ordinal.Principal op o
    ho' : o.IsLimit
    ⊢ LE.le (op a o) o
  -/
  rw [← IsNormal.bsup_eq.{u, u} H ho', bsup_le_iff]
  /-
    a o : Ordinal.{u}
    op : Ordinal.{u} → Ordinal.{u} → Ordinal.{u}
    hao : LT.lt a o
    H : Ordinal.IsNormal (op a)
    ho : Ordinal.Principal op o
    ho' : o.IsLimit
    ⊢ ∀ (i : Ordinal.{u}), LT.lt i o → LE.le (op a i) o
  -/
  exact fun b hbo => (ho hao hbo).le
  /-
    🎉 no goals
  -/


theorem nfp_le_of_principal (hao : a < o) (ho : Principal op o) : nfp (op a) a ≤ o :=
  nfp_le fun n => (ho.iterate_lt hao n).le


/-- We give an explicit construction for a principal ordinal larger or equal than `o`. -/
private theorem principal_nfp_iSup (op : Ordinal → Ordinal → Ordinal) (o : Ordinal) :
    Principal op (nfp (fun x ↦ ⨆ y : Set.Iio x ×ˢ Set.Iio x, succ (op y.1.1 y.1.2)) o) := by
  /-
    op : Ordinal.{u_1} → Ordinal.{u_1} → Ordinal.{u_1}
    o : Ordinal.{u_1}
    ⊢ Ordinal.Principal op (Ordinal.nfp (fun x => iSup fun y => Order.succ (op (↑y …
  -/
  intro a b ha hb
  /-
    op : Ordinal.{u_1} → Ordinal.{u_1} → Ordinal.{u_1}
    o a b : Ordinal.{u_1}
    ha : LT.lt a (Ordinal.nfp (fun x => iSup fun y => Order.succ (op (↑y).1 (↑y).2 …
    hb : LT.lt b (Ordinal.nfp (fun x => iSup fun y => Order.succ (op (↑y).1 (↑y).2 …
    ⊢ LT.lt (op a b) (Ordinal.nfp (fun x => iSup fun y => Order.succ (op (↑y).1 (↑ …
  -/
  rw [lt_nfp] at *
  /-
    op : Ordinal.{u_1} → Ordinal.{u_1} → Ordinal.{u_1}
    o a b : Ordinal.{u_1}
    ha : Exists fun n => LT.lt a (Nat.iterate (fun x => iSup fun y => Order.succ ( …
    hb : Exists fun n => LT.lt b (Nat.iterate (fun x => iSup fun y => Order.succ ( …
    ⊢ Exists fun n => LT.lt (op a b) (Nat.iterate (fun x => iSup fun y => Order.su …
  -/
  obtain ⟨m, ha⟩ := ha
  /-
    case intro
    op : Ordinal.{u_1} → Ordinal.{u_1} → Ordinal.{u_1}
    o a b : Ordinal.{u_1}
    hb : Exists fun n => LT.lt b (Nat.iterate (fun x => iSup fun y => Order.succ ( …
    m : Nat
    ha : LT.lt a (Nat.iterate (fun x => iSup fun y => Order.succ (op (↑y).1 (↑y).2 …
    ⊢ Exists fun n => LT.lt (op a b) (Nat.iterate (fun x => iSup fun y => Order.su …
  -/
  obtain ⟨n, hb⟩ := hb
  obtain h | h := le_total
    ((fun x ↦ ⨆ y : Set.Iio x ×ˢ Set.Iio x, succ (op y.1.1 y.1.2))^[m] o)
    ((fun x ↦ ⨆ y : Set.Iio x ×ˢ Set.Iio x, succ (op y.1.1 y.1.2))^[n] o)
    /-
      case intro.intro.inl
      op : Ordinal.{u_1} → Ordinal.{u_1} → Ordinal.{u_1}
      o a b : Ordinal.{u_1}
      m : Nat
      ha : LT.lt a (Nat.iterate (fun x => iSup fun y => Order.succ (op (↑y).1 (↑y).2 …
      n : Nat
      hb : LT.lt b (Nat.iterate (fun x => iSup fun y => Order.succ (op (↑y).1 (↑y).2 …
      h : LE.le (Nat.iterate (fun x => iSup fun y => Order.succ (op (↑y).1 (↑y).2))  …
      ⊢ Exists fun n => LT.lt (op a b) (Nat.iterate (fun x => iSup fun y => Order.su …
    -/
  · use n + 1
    /-
      case h
      op : Ordinal.{u_1} → Ordinal.{u_1} → Ordinal.{u_1}
      o a b : Ordinal.{u_1}
      m : Nat
      ha : LT.lt a (Nat.iterate (fun x => iSup fun y => Order.succ (op (↑y).1 (↑y).2 …
      n : Nat
      hb : LT.lt b (Nat.iterate (fun x => iSup fun y => Order.succ (op (↑y).1 (↑y).2 …
      h : LE.le (Nat.iterate (fun x => iSup fun y => Order.succ (op (↑y).1 (↑y).2))  …
      ⊢ LT.lt (op a b) (Nat.iterate (fun x => iSup fun y => Order.succ (op (↑y).1 (↑ …
    -/
    rw [Function.iterate_succ']
    /-
      case h
      op : Ordinal.{u_1} → Ordinal.{u_1} → Ordinal.{u_1}
      o a b : Ordinal.{u_1}
      m : Nat
      ha : LT.lt a (Nat.iterate (fun x => iSup fun y => Order.succ (op (↑y).1 (↑y).2 …
      n : Nat
      hb : LT.lt b (Nat.iterate (fun x => iSup fun y => Order.succ (op (↑y).1 (↑y).2 …
      h : LE.le (Nat.iterate (fun x => iSup fun y => Order.succ (op (↑y).1 (↑y).2))  …
      ⊢ LT.lt (op a b) (Function.comp (fun x => iSup fun y => Order.succ (op (↑y).1  …
    -/
    apply (lt_succ _).trans_le
    exact Ordinal.le_iSup (fun y : Set.Iio _ ×ˢ Set.Iio _ ↦ succ (op y.1.1 y.1.2))
      ⟨_, Set.mk_mem_prod (ha.trans_le h) hb⟩
    /-
      case intro.intro.inr
      op : Ordinal.{u_1} → Ordinal.{u_1} → Ordinal.{u_1}
      o a b : Ordinal.{u_1}
      m : Nat
      ha : LT.lt a (Nat.iterate (fun x => iSup fun y => Order.succ (op (↑y).1 (↑y).2 …
      n : Nat
      hb : LT.lt b (Nat.iterate (fun x => iSup fun y => Order.succ (op (↑y).1 (↑y).2 …
      h : LE.le (Nat.iterate (fun x => iSup fun y => Order.succ (op (↑y).1 (↑y).2))  …
      ⊢ Exists fun n => LT.lt (op a b) (Nat.iterate (fun x => iSup fun y => Order.su …
    -/
  · use m + 1
    /-
      case h
      op : Ordinal.{u_1} → Ordinal.{u_1} → Ordinal.{u_1}
      o a b : Ordinal.{u_1}
      m : Nat
      ha : LT.lt a (Nat.iterate (fun x => iSup fun y => Order.succ (op (↑y).1 (↑y).2 …
      n : Nat
      hb : LT.lt b (Nat.iterate (fun x => iSup fun y => Order.succ (op (↑y).1 (↑y).2 …
      h : LE.le (Nat.iterate (fun x => iSup fun y => Order.succ (op (↑y).1 (↑y).2))  …
      ⊢ LT.lt (op a b) (Nat.iterate (fun x => iSup fun y => Order.succ (op (↑y).1 (↑ …
    -/
    rw [Function.iterate_succ']
    /-
      case h
      op : Ordinal.{u_1} → Ordinal.{u_1} → Ordinal.{u_1}
      o a b : Ordinal.{u_1}
      m : Nat
      ha : LT.lt a (Nat.iterate (fun x => iSup fun y => Order.succ (op (↑y).1 (↑y).2 …
      n : Nat
      hb : LT.lt b (Nat.iterate (fun x => iSup fun y => Order.succ (op (↑y).1 (↑y).2 …
      h : LE.le (Nat.iterate (fun x => iSup fun y => Order.succ (op (↑y).1 (↑y).2))  …
      ⊢ LT.lt (op a b) (Function.comp (fun x => iSup fun y => Order.succ (op (↑y).1  …
    -/
    apply (lt_succ _).trans_le
    exact Ordinal.le_iSup (fun y : Set.Iio _ ×ˢ Set.Iio _ ↦ succ (op y.1.1 y.1.2))
      ⟨_, Set.mk_mem_prod ha (hb.trans_le h)⟩


/-- Principal ordinals under any operation are unbounded. -/
theorem not_bddAbove_principal (op : Ordinal → Ordinal → Ordinal) :
    ¬ BddAbove { o | Principal op o } := by
  /-
    op : Ordinal.{u_1} → Ordinal.{u_1} → Ordinal.{u_1}
    ⊢ Not (BddAbove (setOf fun o => Ordinal.Principal op o))
  -/
  rintro ⟨a, ha⟩
  /-
    case intro
    op : Ordinal.{u_1} → Ordinal.{u_1} → Ordinal.{u_1}
    a : Ordinal.{u_1}
    ha : Membership.mem (upperBounds (setOf fun o => Ordinal.Principal op o)) a
    ⊢ False
  -/
  exact ((le_nfp _ _).trans (ha (principal_nfp_iSup op (succ a)))).not_lt (lt_succ a)
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided." (since := "2024-10-11")]
theorem principal_nfp_blsub₂ (op : Ordinal → Ordinal → Ordinal) (o : Ordinal) :
    Principal op (nfp (fun o' => blsub₂.{u, u, u} o' o' (@fun a _ b _ => op a b)) o) := by
  /-
    op : Ordinal.{u} → Ordinal.{u} → Ordinal.{u}
    o : Ordinal.{u}
    ⊢ Ordinal.Principal op (Ordinal.nfp (fun o' => o'.blsub₂ o' fun a x b x => op  …
  -/
  intro a b ha hb
  /-
    op : Ordinal.{u} → Ordinal.{u} → Ordinal.{u}
    o a b : Ordinal.{u}
    ha : LT.lt a (Ordinal.nfp (fun o' => o'.blsub₂ o' fun a x b x => op a b) o)
    hb : LT.lt b (Ordinal.nfp (fun o' => o'.blsub₂ o' fun a x b x => op a b) o)
    ⊢ LT.lt (op a b) (Ordinal.nfp (fun o' => o'.blsub₂ o' fun a x b x => op a b) o)
  -/
  rw [lt_nfp] at *
  /-
    op : Ordinal.{u} → Ordinal.{u} → Ordinal.{u}
    o a b : Ordinal.{u}
    ha : Exists fun n => LT.lt a (Nat.iterate (fun o' => o'.blsub₂ o' fun a x b x  …
    hb : Exists fun n => LT.lt b (Nat.iterate (fun o' => o'.blsub₂ o' fun a x b x  …
    ⊢ Exists fun n => LT.lt (op a b) (Nat.iterate (fun o' => o'.blsub₂ o' fun a x  …
  -/
  cases' ha with m hm
  /-
    case intro
    op : Ordinal.{u} → Ordinal.{u} → Ordinal.{u}
    o a b : Ordinal.{u}
    hb : Exists fun n => LT.lt b (Nat.iterate (fun o' => o'.blsub₂ o' fun a x b x  …
    m : Nat
    hm : LT.lt a (Nat.iterate (fun o' => o'.blsub₂ o' fun a x b x => op a b) m o)
    ⊢ Exists fun n => LT.lt (op a b) (Nat.iterate (fun o' => o'.blsub₂ o' fun a x  …
  -/
  cases' hb with n hn
  cases' le_total
    ((fun o' => blsub₂.{u, u, u} o' o' (@fun a _ b _ => op a b))^[m] o)
    ((fun o' => blsub₂.{u, u, u} o' o' (@fun a _ b _ => op a b))^[n] o) with h h
    /-
      case intro.intro.inl
      op : Ordinal.{u} → Ordinal.{u} → Ordinal.{u}
      o a b : Ordinal.{u}
      m : Nat
      hm : LT.lt a (Nat.iterate (fun o' => o'.blsub₂ o' fun a x b x => op a b) m o)
      n : Nat
      hn : LT.lt b (Nat.iterate (fun o' => o'.blsub₂ o' fun a x b x => op a b) n o)
      h : LE.le (Nat.iterate (fun o' => o'.blsub₂ o' fun a x b x => op a b) m o) (Na …
      ⊢ Exists fun n => LT.lt (op a b) (Nat.iterate (fun o' => o'.blsub₂ o' fun a x  …
    -/
  · use n + 1
    /-
      case h
      op : Ordinal.{u} → Ordinal.{u} → Ordinal.{u}
      o a b : Ordinal.{u}
      m : Nat
      hm : LT.lt a (Nat.iterate (fun o' => o'.blsub₂ o' fun a x b x => op a b) m o)
      n : Nat
      hn : LT.lt b (Nat.iterate (fun o' => o'.blsub₂ o' fun a x b x => op a b) n o)
      h : LE.le (Nat.iterate (fun o' => o'.blsub₂ o' fun a x b x => op a b) m o) (Na …
      ⊢ LT.lt (op a b) (Nat.iterate (fun o' => o'.blsub₂ o' fun a x b x => op a b) ( …
    -/
    rw [Function.iterate_succ']
    /-
      case h
      op : Ordinal.{u} → Ordinal.{u} → Ordinal.{u}
      o a b : Ordinal.{u}
      m : Nat
      hm : LT.lt a (Nat.iterate (fun o' => o'.blsub₂ o' fun a x b x => op a b) m o)
      n : Nat
      hn : LT.lt b (Nat.iterate (fun o' => o'.blsub₂ o' fun a x b x => op a b) n o)
      h : LE.le (Nat.iterate (fun o' => o'.blsub₂ o' fun a x b x => op a b) m o) (Na …
      ⊢ LT.lt (op a b) (Function.comp (fun o' => o'.blsub₂ o' fun a x b x => op a b) …
    -/
    exact lt_blsub₂ (@fun a _ b _ => op a b) (hm.trans_le h) hn
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr
      op : Ordinal.{u} → Ordinal.{u} → Ordinal.{u}
      o a b : Ordinal.{u}
      m : Nat
      hm : LT.lt a (Nat.iterate (fun o' => o'.blsub₂ o' fun a x b x => op a b) m o)
      n : Nat
      hn : LT.lt b (Nat.iterate (fun o' => o'.blsub₂ o' fun a x b x => op a b) n o)
      h : LE.le (Nat.iterate (fun o' => o'.blsub₂ o' fun a x b x => op a b) n o) (Na …
      ⊢ Exists fun n => LT.lt (op a b) (Nat.iterate (fun o' => o'.blsub₂ o' fun a x  …
    -/
  · use m + 1
    /-
      case h
      op : Ordinal.{u} → Ordinal.{u} → Ordinal.{u}
      o a b : Ordinal.{u}
      m : Nat
      hm : LT.lt a (Nat.iterate (fun o' => o'.blsub₂ o' fun a x b x => op a b) m o)
      n : Nat
      hn : LT.lt b (Nat.iterate (fun o' => o'.blsub₂ o' fun a x b x => op a b) n o)
      h : LE.le (Nat.iterate (fun o' => o'.blsub₂ o' fun a x b x => op a b) n o) (Na …
      ⊢ LT.lt (op a b) (Nat.iterate (fun o' => o'.blsub₂ o' fun a x b x => op a b) ( …
    -/
    rw [Function.iterate_succ']
    /-
      case h
      op : Ordinal.{u} → Ordinal.{u} → Ordinal.{u}
      o a b : Ordinal.{u}
      m : Nat
      hm : LT.lt a (Nat.iterate (fun o' => o'.blsub₂ o' fun a x b x => op a b) m o)
      n : Nat
      hn : LT.lt b (Nat.iterate (fun o' => o'.blsub₂ o' fun a x b x => op a b) n o)
      h : LE.le (Nat.iterate (fun o' => o'.blsub₂ o' fun a x b x => op a b) n o) (Na …
      ⊢ LT.lt (op a b) (Function.comp (fun o' => o'.blsub₂ o' fun a x b x => op a b) …
    -/
    exact lt_blsub₂ (@fun a _ b _ => op a b) hm (hn.trans_le h)
    /-
      🎉 no goals
    -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided." (since := "2024-10-11")]
theorem unbounded_principal (op : Ordinal → Ordinal → Ordinal) :
    Set.Unbounded (· < ·) { o | Principal op o } := fun o =>
  ⟨_, principal_nfp_blsub₂ op o, (le_nfp _ o).not_lt⟩


theorem principal_add_one : Principal (· + ·) 1 :=
  principal_one_iff.2 <| zero_add 0


theorem principal_add_of_le_one (ho : o ≤ 1) : Principal (· + ·) o := by
  /-
    o : Ordinal.{u}
    ho : LE.le o 1
    ⊢ Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) o
  -/
  rcases le_one_iff.1 ho with (rfl | rfl)
    /-
      case inl
      ho : LE.le 0 1
      ⊢ Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) 0
    -/
  · exact principal_zero
    /-
      🎉 no goals
    -/
    /-
      case inr
      ho : LE.le 1 1
      ⊢ Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) 1
    -/
  · exact principal_add_one
    /-
      🎉 no goals
    -/


theorem isLimit_of_principal_add (ho₁ : 1 < o) (ho : Principal (· + ·) o) : o.IsLimit := by
  /-
    o : Ordinal.{u}
    ho₁ : LT.lt 1 o
    ho : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) o
    ⊢ o.IsLimit
  -/
  rw [isLimit_iff, isSuccPrelimit_iff_succ_lt]
  /-
    o : Ordinal.{u}
    ho₁ : LT.lt 1 o
    ho : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) o
    ⊢ And (Ne o 0) (∀ (a : Ordinal.{u}), LT.lt a o → LT.lt (Order.succ a) o)
  -/
  exact ⟨ho₁.ne_bot, fun _ ha ↦ ho ha ho₁⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-16")]
alias principal_add_isLimit := isLimit_of_principal_add


theorem principal_add_iff_add_left_eq_self : Principal (· + ·) o ↔ ∀ a < o, a + o = o := by
  /-
    o : Ordinal.{u}
    ⊢ Iff (Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) o) (∀ (a : Ordinal.{u} …
  -/
  refine ⟨fun ho a hao => ?_, fun h a b hao hbo => ?_⟩
    /-
      case refine_1
      o : Ordinal.{u}
      ho : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) o
      a : Ordinal.{u}
      hao : LT.lt a o
      ⊢ Eq (HAdd.hAdd a o) o
    -/
  · cases' lt_or_le 1 o with ho₁ ho₁
      /-
        case refine_1.inl
        o : Ordinal.{u}
        ho : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) o
        a : Ordinal.{u}
        hao : LT.lt a o
        ho₁ : LT.lt 1 o
        ⊢ Eq (HAdd.hAdd a o) o
      -/
    · exact op_eq_self_of_principal hao (isNormal_add_right a) ho (isLimit_of_principal_add ho₁ ho)
      /-
        🎉 no goals
      -/
      /-
        case refine_1.inr
        o : Ordinal.{u}
        ho : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) o
        a : Ordinal.{u}
        hao : LT.lt a o
        ho₁ : LE.le o 1
        ⊢ Eq (HAdd.hAdd a o) o
      -/
    · rcases le_one_iff.1 ho₁ with (rfl | rfl)
        /-
          case refine_1.inr.inl
          a : Ordinal.{u}
          ho : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) 0
          hao : LT.lt a 0
          ho₁ : LE.le 0 1
          ⊢ Eq (HAdd.hAdd a 0) 0
        -/
      · exact (Ordinal.not_lt_zero a hao).elim
        /-
          🎉 no goals
        -/
        /-
          case refine_1.inr.inr
          a : Ordinal.{u}
          ho : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) 1
          hao : LT.lt a 1
          ho₁ : LE.le 1 1
          ⊢ Eq (HAdd.hAdd a 1) 1
        -/
      · rw [lt_one_iff_zero] at hao
        /-
          case refine_1.inr.inr
          a : Ordinal.{u}
          ho : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) 1
          hao : Eq a 0
          ho₁ : LE.le 1 1
          ⊢ Eq (HAdd.hAdd a 1) 1
        -/
        rw [hao, zero_add]
        /-
          🎉 no goals
        -/
    /-
      case refine_2
      o : Ordinal.{u}
      h : ∀ (a : Ordinal.{u}), LT.lt a o → Eq (HAdd.hAdd a o) o
      a b : Ordinal.{u}
      hao : LT.lt a o
      hbo : LT.lt b o
      ⊢ LT.lt ((fun x1 x2 => HAdd.hAdd x1 x2) a b) o
    -/
  · rw [← h a hao]
    /-
      case refine_2
      o : Ordinal.{u}
      h : ∀ (a : Ordinal.{u}), LT.lt a o → Eq (HAdd.hAdd a o) o
      a b : Ordinal.{u}
      hao : LT.lt a o
      hbo : LT.lt b o
      ⊢ LT.lt ((fun x1 x2 => HAdd.hAdd x1 x2) a b) (HAdd.hAdd a o)
    -/
    exact (isNormal_add_right a).strictMono hbo
    /-
      🎉 no goals
    -/


theorem exists_lt_add_of_not_principal_add (ha : ¬ Principal (· + ·) a) :
    ∃ b < a, ∃ c < a, b + c = a := by
  /-
    a : Ordinal.{u}
    ha : Not (Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) a)
    ⊢ Exists fun b => And (LT.lt b a) (Exists fun c => And (LT.lt c a) (Eq (HAdd.h …
  -/
  rw [not_principal_iff] at ha
  /-
    a : Ordinal.{u}
    ha : Exists fun a_1 => And (LT.lt a_1 a) (Exists fun b => And (LT.lt b a) (LE. …
    ⊢ Exists fun b => And (LT.lt b a) (Exists fun c => And (LT.lt c a) (Eq (HAdd.h …
  -/
  rcases ha with ⟨b, hb, c, hc, H⟩
  refine
    ⟨b, hb, _, lt_of_le_of_ne (sub_le_self a b) fun hab => ?_, Ordinal.add_sub_cancel_of_le hb.le⟩
  /-
    case intro.intro.intro.intro
    a b : Ordinal.{u}
    hb : LT.lt b a
    c : Ordinal.{u}
    hc : LT.lt c a
    H : LE.le a (HAdd.hAdd b c)
    hab : Eq (HSub.hSub a b) a
    ⊢ False
  -/
  rw [← sub_le, hab] at H
  /-
    case intro.intro.intro.intro
    a b : Ordinal.{u}
    hb : LT.lt b a
    c : Ordinal.{u}
    hc : LT.lt c a
    H : LE.le a c
    hab : Eq (HSub.hSub a b) a
    ⊢ False
  -/
  exact H.not_lt hc
  /-
    🎉 no goals
  -/


theorem principal_add_iff_add_lt_ne_self : Principal (· + ·) a ↔ ∀ b < a, ∀ c < a, b + c ≠ a :=
  ⟨fun ha _ hb _ hc => (ha hb hc).ne, fun H => by
    /-
      a : Ordinal.{u}
      H : ∀ (b : Ordinal.{u}), LT.lt b a → ∀ (c : Ordinal.{u}), LT.lt c a → Ne (HAdd …
      ⊢ Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) a
    -/
    by_contra! ha
    /-
      a : Ordinal.{u}
      H : ∀ (b : Ordinal.{u}), LT.lt b a → ∀ (c : Ordinal.{u}), LT.lt c a → Ne (HAdd …
      ha : Not (Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) a)
      ⊢ False
    -/
    rcases exists_lt_add_of_not_principal_add ha with ⟨b, hb, c, hc, rfl⟩
    /-
      case intro.intro.intro.intro
      b c : Ordinal.{u}
      H : ∀ (b_1 : Ordinal.{u}), LT.lt b_1 (HAdd.hAdd b c) → ∀ (c_1 : Ordinal.{u}),  …
      ha : Not (Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) (HAdd.hAdd b c))
      hb : LT.lt b (HAdd.hAdd b c)
      hc : LT.lt c (HAdd.hAdd b c)
      ⊢ False
    -/
    exact (H b hb c hc).irrefl⟩
    /-
      🎉 no goals
    -/


theorem principal_add_omega0 : Principal (· + ·) ω :=
  principal_add_iff_add_left_eq_self.2 fun _ => add_omega0


@[deprecated (since := "2024-09-30")]
alias principal_add_omega := principal_add_omega0


theorem add_omega0_opow (h : a < ω ^ b) : a + ω ^ b = ω ^ b := by
  /-
    a b : Ordinal.{u}
    h : LT.lt a (HPow.hPow Ordinal.omega0 b)
    ⊢ Eq (HAdd.hAdd a (HPow.hPow Ordinal.omega0 b)) (HPow.hPow Ordinal.omega0 b)
  -/
  refine le_antisymm ?_ (le_add_left _ a)
  /-
    a b : Ordinal.{u}
    h : LT.lt a (HPow.hPow Ordinal.omega0 b)
    ⊢ LE.le (HAdd.hAdd a (HPow.hPow Ordinal.omega0 b)) (HPow.hPow Ordinal.omega0 b)
  -/
  induction' b using limitRecOn with b _ b l IH
    /-
      case H₁
      a b : Ordinal.{u}
      h : LT.lt a (HPow.hPow Ordinal.omega0 0)
      ⊢ LE.le (HAdd.hAdd a (HPow.hPow Ordinal.omega0 0)) (HPow.hPow Ordinal.omega0 0)
    -/
  · rw [opow_zero, ← succ_zero, lt_succ_iff, Ordinal.le_zero] at h
    /-
      case H₁
      a b : Ordinal.{u}
      h : Eq a 0
      ⊢ LE.le (HAdd.hAdd a (HPow.hPow Ordinal.omega0 0)) (HPow.hPow Ordinal.omega0 0)
    -/
    rw [h, zero_add]
    /-
      🎉 no goals
    -/
    /-
      case H₂
      a b✝ b : Ordinal.{u}
      a✝ : LT.lt a (HPow.hPow Ordinal.omega0 b) → LE.le (HAdd.hAdd a (HPow.hPow Ordi …
      h : LT.lt a (HPow.hPow Ordinal.omega0 (Order.succ b))
      ⊢ LE.le (HAdd.hAdd a (HPow.hPow Ordinal.omega0 (Order.succ b))) (HPow.hPow Ord …
    -/
  · rw [opow_succ] at h
    /-
      case H₂
      a b✝ b : Ordinal.{u}
      a✝ : LT.lt a (HPow.hPow Ordinal.omega0 b) → LE.le (HAdd.hAdd a (HPow.hPow Ordi …
      h : LT.lt a (HMul.hMul (HPow.hPow Ordinal.omega0 b) Ordinal.omega0)
      ⊢ LE.le (HAdd.hAdd a (HPow.hPow Ordinal.omega0 (Order.succ b))) (HPow.hPow Ord …
    -/
    rcases (lt_mul_of_limit isLimit_omega0).1 h with ⟨x, xo, ax⟩
    /-
      case H₂.intro.intro
      a b✝ b : Ordinal.{u}
      a✝ : LT.lt a (HPow.hPow Ordinal.omega0 b) → LE.le (HAdd.hAdd a (HPow.hPow Ordi …
      h : LT.lt a (HMul.hMul (HPow.hPow Ordinal.omega0 b) Ordinal.omega0)
      x : Ordinal.{u}
      xo : LT.lt x Ordinal.omega0
      ax : LT.lt a (HMul.hMul (HPow.hPow Ordinal.omega0 b) x)
      ⊢ LE.le (HAdd.hAdd a (HPow.hPow Ordinal.omega0 (Order.succ b))) (HPow.hPow Ord …
    -/
    apply (add_le_add_right ax.le _).trans
    /-
      case H₂.intro.intro
      a b✝ b : Ordinal.{u}
      a✝ : LT.lt a (HPow.hPow Ordinal.omega0 b) → LE.le (HAdd.hAdd a (HPow.hPow Ordi …
      h : LT.lt a (HMul.hMul (HPow.hPow Ordinal.omega0 b) Ordinal.omega0)
      x : Ordinal.{u}
      xo : LT.lt x Ordinal.omega0
      ax : LT.lt a (HMul.hMul (HPow.hPow Ordinal.omega0 b) x)
      ⊢ LE.le (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 b) x) (HPow.hPow Ordin …
    -/
    rw [opow_succ, ← mul_add, add_omega0 xo]
    /-
      🎉 no goals
    -/
    /-
      case H₃
      a b✝ b : Ordinal.{u}
      l : b.IsLimit
      IH : ∀ (o' : Ordinal.{u}), LT.lt o' b → LT.lt a (HPow.hPow Ordinal.omega0 o')  …
      h : LT.lt a (HPow.hPow Ordinal.omega0 b)
      ⊢ LE.le (HAdd.hAdd a (HPow.hPow Ordinal.omega0 b)) (HPow.hPow Ordinal.omega0 b)
    -/
  · rcases (lt_opow_of_limit omega0_ne_zero l).1 h with ⟨x, xb, ax⟩
    /-
      case H₃.intro.intro
      a b✝ b : Ordinal.{u}
      l : b.IsLimit
      IH : ∀ (o' : Ordinal.{u}), LT.lt o' b → LT.lt a (HPow.hPow Ordinal.omega0 o')  …
      h : LT.lt a (HPow.hPow Ordinal.omega0 b)
      x : Ordinal.{u}
      xb : LT.lt x b
      ax : LT.lt a (HPow.hPow Ordinal.omega0 x)
      ⊢ LE.le (HAdd.hAdd a (HPow.hPow Ordinal.omega0 b)) (HPow.hPow Ordinal.omega0 b)
    -/
    apply (((isNormal_add_right a).trans <| isNormal_opow one_lt_omega0).limit_le l).2
    /-
      case H₃.intro.intro
      a b✝ b : Ordinal.{u}
      l : b.IsLimit
      IH : ∀ (o' : Ordinal.{u}), LT.lt o' b → LT.lt a (HPow.hPow Ordinal.omega0 o')  …
      h : LT.lt a (HPow.hPow Ordinal.omega0 b)
      x : Ordinal.{u}
      xb : LT.lt x b
      ax : LT.lt a (HPow.hPow Ordinal.omega0 x)
      ⊢ ∀ (b_1 : Ordinal.{u}), LT.lt b_1 b → LE.le (Function.comp (fun x => HAdd.hAd …
    -/
    intro y yb
    calc a + ω ^ y ≤ a + ω ^ max x y :=
      add_le_add_left (opow_le_opow_right omega0_pos (le_max_right x y)) _
    _ ≤ ω ^ max x y :=
      IH _ (max_lt xb yb) <| ax.trans_le <| opow_le_opow_right omega0_pos <| le_max_left x y
    _ ≤ ω ^ b :=
      opow_le_opow_right omega0_pos <| (max_lt xb yb).le


@[deprecated (since := "2024-09-30")]
alias add_omega_opow := add_omega0_opow


theorem principal_add_omega0_opow (o : Ordinal) : Principal (· + ·) (ω ^ o) :=
  principal_add_iff_add_left_eq_self.2 fun _ => add_omega0_opow


@[deprecated (since := "2024-09-30")]
alias principal_add_omega_opow := principal_add_omega0_opow


/-- The main characterization theorem for additive principal ordinals. -/
theorem principal_add_iff_zero_or_omega0_opow :
    Principal (· + ·) o ↔ o = 0 ∨ o ∈ Set.range (ω ^ · : Ordinal → Ordinal) := by
  /-
    o : Ordinal.{u}
    ⊢ Iff (Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) o) (Or (Eq o 0) (Membe …
  -/
  rcases eq_or_ne o 0 with (rfl | ho)
    /-
      case inl
      ⊢ Iff (Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) 0) (Or (Eq 0 0) (Membe …
    -/
  · simp only [principal_zero, Or.inl]
    /-
      🎉 no goals
    -/
    /-
      case inr
      o : Ordinal.{u}
      ho : Ne o 0
      ⊢ Iff (Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) o) (Or (Eq o 0) (Membe …
    -/
  · rw [principal_add_iff_add_left_eq_self]
    /-
      case inr
      o : Ordinal.{u}
      ho : Ne o 0
      ⊢ Iff (∀ (a : Ordinal.{u}), LT.lt a o → Eq (HAdd.hAdd a o) o) (Or (Eq o 0) (Me …
    -/
    simp only [ho, false_or]
    refine
      ⟨fun H => ⟨_, ((lt_or_eq_of_le (opow_log_le_self _ ho)).resolve_left fun h => ?_)⟩,
        fun ⟨b, e⟩ => e.symm ▸ fun a => add_omega0_opow⟩
    /-
      case inr
      o : Ordinal.{u}
      ho : Ne o 0
      H : ∀ (a : Ordinal.{u}), LT.lt a o → Eq (HAdd.hAdd a o) o
      h : LT.lt (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 o)) o
      ⊢ False
    -/
    have := H _ h
    /-
      case inr
      o : Ordinal.{u}
      ho : Ne o 0
      H : ∀ (a : Ordinal.{u}), LT.lt a o → Eq (HAdd.hAdd a o) o
      h : LT.lt (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 o)) o
      this : Eq (HAdd.hAdd (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 o)) …
      ⊢ False
    -/
    have := lt_opow_succ_log_self one_lt_omega0 o
    /-
      case inr
      o : Ordinal.{u}
      ho : Ne o 0
      H : ∀ (a : Ordinal.{u}), LT.lt a o → Eq (HAdd.hAdd a o) o
      h : LT.lt (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 o)) o
      this✝ : Eq (HAdd.hAdd (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 o) …
      this : LT.lt o (HPow.hPow Ordinal.omega0 (Order.succ (Ordinal.log Ordinal.omeg …
      ⊢ False
    -/
    rw [opow_succ, lt_mul_of_limit isLimit_omega0] at this
    /-
      case inr
      o : Ordinal.{u}
      ho : Ne o 0
      H : ∀ (a : Ordinal.{u}), LT.lt a o → Eq (HAdd.hAdd a o) o
      h : LT.lt (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 o)) o
      this✝ : Eq (HAdd.hAdd (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 o) …
      this : Exists fun c' => And (LT.lt c' Ordinal.omega0) (LT.lt o (HMul.hMul (HPo …
      ⊢ False
    -/
    rcases this with ⟨a, ao, h'⟩
    /-
      case inr.intro.intro
      o : Ordinal.{u}
      ho : Ne o 0
      H : ∀ (a : Ordinal.{u}), LT.lt a o → Eq (HAdd.hAdd a o) o
      h : LT.lt (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 o)) o
      this : Eq (HAdd.hAdd (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 o)) …
      a : Ordinal.{u}
      ao : LT.lt a Ordinal.omega0
      h' : LT.lt o (HMul.hMul (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0  …
      ⊢ False
    -/
    rcases lt_omega0.1 ao with ⟨n, rfl⟩
    /-
      case inr.intro.intro.intro
      o : Ordinal.{u}
      ho : Ne o 0
      H : ∀ (a : Ordinal.{u}), LT.lt a o → Eq (HAdd.hAdd a o) o
      h : LT.lt (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 o)) o
      this : Eq (HAdd.hAdd (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 o)) …
      n : Nat
      ao : LT.lt (↑n) Ordinal.omega0
      h' : LT.lt o (HMul.hMul (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0  …
      ⊢ False
    -/
    clear ao
    /-
      case inr.intro.intro.intro
      o : Ordinal.{u}
      ho : Ne o 0
      H : ∀ (a : Ordinal.{u}), LT.lt a o → Eq (HAdd.hAdd a o) o
      h : LT.lt (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 o)) o
      this : Eq (HAdd.hAdd (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 o)) …
      n : Nat
      h' : LT.lt o (HMul.hMul (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0  …
      ⊢ False
    -/
    revert h'
    /-
      case inr.intro.intro.intro
      o : Ordinal.{u}
      ho : Ne o 0
      H : ∀ (a : Ordinal.{u}), LT.lt a o → Eq (HAdd.hAdd a o) o
      h : LT.lt (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 o)) o
      this : Eq (HAdd.hAdd (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 o)) …
      n : Nat
      ⊢ LT.lt o (HMul.hMul (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 o)) …
    -/
    apply not_lt_of_le
    suffices e : ω ^ log ω o * n + o = o by
      simpa only [e] using le_add_right (ω ^ log ω o * ↑n) o
    /-
      case inr.intro.intro.intro.hab
      o : Ordinal.{u}
      ho : Ne o 0
      H : ∀ (a : Ordinal.{u}), LT.lt a o → Eq (HAdd.hAdd a o) o
      h : LT.lt (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 o)) o
      this : Eq (HAdd.hAdd (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 o)) …
      n : Nat
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omeg …
    -/
    induction' n with n IH
      /-
        case inr.intro.intro.intro.hab.zero
        o : Ordinal.{u}
        ho : Ne o 0
        H : ∀ (a : Ordinal.{u}), LT.lt a o → Eq (HAdd.hAdd a o) o
        h : LT.lt (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 o)) o
        this : Eq (HAdd.hAdd (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 o)) …
        ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omeg …
      -/
    · simp [Nat.cast_zero, mul_zero, zero_add]
      /-
        🎉 no goals
      -/
      /-
        case inr.intro.intro.intro.hab.succ
        o : Ordinal.{u}
        ho : Ne o 0
        H : ∀ (a : Ordinal.{u}), LT.lt a o → Eq (HAdd.hAdd a o) o
        h : LT.lt (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 o)) o
        this : Eq (HAdd.hAdd (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 o)) …
        n : Nat
        IH : Eq (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.o …
        ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omeg …
      -/
    · simp only [Nat.cast_succ, mul_add_one, add_assoc, this, IH]
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-09-30")]
alias principal_add_iff_zero_or_omega_opow := principal_add_iff_zero_or_omega0_opow


theorem principal_add_opow_of_principal_add {a} (ha : Principal (· + ·) a) (b : Ordinal) :
    Principal (· + ·) (a ^ b) := by
  /-
    a : Ordinal.{u_1}
    ha : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) a
    b : Ordinal.{u_1}
    ⊢ Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) (HPow.hPow a b)
  -/
  rcases principal_add_iff_zero_or_omega0_opow.1 ha with (rfl | ⟨c, rfl⟩)
    /-
      case inl
      b : Ordinal.{u_1}
      ha : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) 0
      ⊢ Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) (HPow.hPow 0 b)
    -/
  · rcases eq_or_ne b 0 with (rfl | hb)
      /-
        case inl.inl
        ha : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) 0
        ⊢ Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) (HPow.hPow 0 0)
      -/
    · rw [opow_zero]
      /-
        case inl.inl
        ha : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) 0
        ⊢ Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) 1
      -/
      exact principal_add_one
      /-
        🎉 no goals
      -/
      /-
        case inl.inr
        b : Ordinal.{u_1}
        ha : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) 0
        hb : Ne b 0
        ⊢ Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) (HPow.hPow 0 b)
      -/
    · rwa [zero_opow hb]
      /-
        🎉 no goals
      -/
    /-
      case inr.intro
      b c : Ordinal.{u_1}
      ha : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) ((fun x => HPow.hPow Ord …
      ⊢ Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) (HPow.hPow ((fun x => HPow. …
    -/
  · rw [← opow_mul]
    /-
      case inr.intro
      b c : Ordinal.{u_1}
      ha : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) ((fun x => HPow.hPow Ord …
      ⊢ Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) (HPow.hPow Ordinal.omega0 ( …
    -/
    exact principal_add_omega0_opow _
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-16")]
alias opow_principal_add_of_principal_add := principal_add_opow_of_principal_add


theorem add_absorp (h₁ : a < ω ^ b) (h₂ : ω ^ b ≤ c) : a + c = c := by
  /-
    a b c : Ordinal.{u}
    h₁ : LT.lt a (HPow.hPow Ordinal.omega0 b)
    h₂ : LE.le (HPow.hPow Ordinal.omega0 b) c
    ⊢ Eq (HAdd.hAdd a c) c
  -/
  rw [← Ordinal.add_sub_cancel_of_le h₂, ← add_assoc, add_omega0_opow h₁]
  /-
    🎉 no goals
  -/


theorem principal_add_mul_of_principal_add (a : Ordinal.{u}) {b : Ordinal.{u}} (hb₁ : b ≠ 1)
    (hb : Principal (· + ·) b) : Principal (· + ·) (a * b) := by
  /-
    a b : Ordinal.{u}
    hb₁ : Ne b 1
    hb : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) b
    ⊢ Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) (HMul.hMul a b)
  -/
  rcases eq_zero_or_pos a with (rfl | _)
    /-
      case inl
      b : Ordinal.{u}
      hb₁ : Ne b 1
      hb : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) b
      ⊢ Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) (HMul.hMul 0 b)
    -/
  · rw [zero_mul]
    /-
      case inl
      b : Ordinal.{u}
      hb₁ : Ne b 1
      hb : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) b
      ⊢ Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) 0
    -/
    exact principal_zero
    /-
      🎉 no goals
    -/
    /-
      case inr
      a b : Ordinal.{u}
      hb₁ : Ne b 1
      hb : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) b
      h✝ : LT.lt 0 a
      ⊢ Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) (HMul.hMul a b)
    -/
  · rcases eq_zero_or_pos b with (rfl | hb₁')
      /-
        case inr.inl
        a : Ordinal.{u}
        h✝ : LT.lt 0 a
        hb₁ : Ne 0 1
        hb : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) 0
        ⊢ Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) (HMul.hMul a 0)
      -/
    · rw [mul_zero]
      /-
        case inr.inl
        a : Ordinal.{u}
        h✝ : LT.lt 0 a
        hb₁ : Ne 0 1
        hb : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) 0
        ⊢ Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) 0
      -/
      exact principal_zero
      /-
        🎉 no goals
      -/
      /-
        case inr.inr
        a b : Ordinal.{u}
        hb₁ : Ne b 1
        hb : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) b
        h✝ : LT.lt 0 a
        hb₁' : LT.lt 0 b
        ⊢ Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) (HMul.hMul a b)
      -/
    · rw [← succ_le_iff, succ_zero] at hb₁'
      /-
        case inr.inr
        a b : Ordinal.{u}
        hb₁ : Ne b 1
        hb : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) b
        h✝ : LT.lt 0 a
        hb₁' : LE.le 1 b
        ⊢ Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) (HMul.hMul a b)
      -/
      intro c d hc hd
      /-
        case inr.inr
        a b : Ordinal.{u}
        hb₁ : Ne b 1
        hb : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) b
        h✝ : LT.lt 0 a
        hb₁' : LE.le 1 b
        c d : Ordinal.{u}
        hc : LT.lt c (HMul.hMul a b)
        hd : LT.lt d (HMul.hMul a b)
        ⊢ LT.lt ((fun x1 x2 => HAdd.hAdd x1 x2) c d) (HMul.hMul a b)
      -/
      rw [lt_mul_of_limit (isLimit_of_principal_add (lt_of_le_of_ne hb₁' hb₁.symm) hb)] at *
      /-
        case inr.inr
        a b : Ordinal.{u}
        hb₁ : Ne b 1
        hb : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) b
        h✝ : LT.lt 0 a
        hb₁' : LE.le 1 b
        c d : Ordinal.{u}
        hc : Exists fun c' => And (LT.lt c' b) (LT.lt c (HMul.hMul a c'))
        hd : Exists fun c' => And (LT.lt c' b) (LT.lt d (HMul.hMul a c'))
        ⊢ Exists fun c' => And (LT.lt c' b) (LT.lt ((fun x1 x2 => HAdd.hAdd x1 x2) c d …
      -/
      rcases hc with ⟨x, hx, hx'⟩
      /-
        case inr.inr.intro.intro
        a b : Ordinal.{u}
        hb₁ : Ne b 1
        hb : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) b
        h✝ : LT.lt 0 a
        hb₁' : LE.le 1 b
        c d : Ordinal.{u}
        hd : Exists fun c' => And (LT.lt c' b) (LT.lt d (HMul.hMul a c'))
        x : Ordinal.{u}
        hx : LT.lt x b
        hx' : LT.lt c (HMul.hMul a x)
        ⊢ Exists fun c' => And (LT.lt c' b) (LT.lt ((fun x1 x2 => HAdd.hAdd x1 x2) c d …
      -/
      rcases hd with ⟨y, hy, hy'⟩
      /-
        case inr.inr.intro.intro.intro.intro
        a b : Ordinal.{u}
        hb₁ : Ne b 1
        hb : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) b
        h✝ : LT.lt 0 a
        hb₁' : LE.le 1 b
        c d x : Ordinal.{u}
        hx : LT.lt x b
        hx' : LT.lt c (HMul.hMul a x)
        y : Ordinal.{u}
        hy : LT.lt y b
        hy' : LT.lt d (HMul.hMul a y)
        ⊢ Exists fun c' => And (LT.lt c' b) (LT.lt ((fun x1 x2 => HAdd.hAdd x1 x2) c d …
      -/
      use x + y, hb hx hy
      /-
        case right
        a b : Ordinal.{u}
        hb₁ : Ne b 1
        hb : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) b
        h✝ : LT.lt 0 a
        hb₁' : LE.le 1 b
        c d x : Ordinal.{u}
        hx : LT.lt x b
        hx' : LT.lt c (HMul.hMul a x)
        y : Ordinal.{u}
        hy : LT.lt y b
        hy' : LT.lt d (HMul.hMul a y)
        ⊢ LT.lt ((fun x1 x2 => HAdd.hAdd x1 x2) c d) (HMul.hMul a (HAdd.hAdd x y))
      -/
      rw [mul_add]
      /-
        case right
        a b : Ordinal.{u}
        hb₁ : Ne b 1
        hb : Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) b
        h✝ : LT.lt 0 a
        hb₁' : LE.le 1 b
        c d x : Ordinal.{u}
        hx : LT.lt x b
        hx' : LT.lt c (HMul.hMul a x)
        y : Ordinal.{u}
        hy : LT.lt y b
        hy' : LT.lt d (HMul.hMul a y)
        ⊢ LT.lt ((fun x1 x2 => HAdd.hAdd x1 x2) c d) (HAdd.hAdd (HMul.hMul a x) (HMul. …
      -/
      exact Left.add_lt_add hx' hy'
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-10-16")]
alias mul_principal_add_is_principal_add := principal_add_mul_of_principal_add


theorem principal_mul_one : Principal (· * ·) 1 := by
  /-
    ⊢ Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) 1
  -/
  rw [principal_one_iff]
  /-
    ⊢ Eq (HMul.hMul 0 0) 0
  -/
  exact zero_mul _
  /-
    🎉 no goals
  -/


theorem principal_mul_two : Principal (· * ·) 2 := by
  /-
    ⊢ Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) 2
  -/
  intro a b ha hb
  /-
    a b : Ordinal.{u_1}
    ha : LT.lt a 2
    hb : LT.lt b 2
    ⊢ LT.lt ((fun x1 x2 => HMul.hMul x1 x2) a b) 2
  -/
  rw [← succ_one, lt_succ_iff] at *
  /-
    a b : Ordinal.{u_1}
    ha : LE.le a 1
    hb : LE.le b 1
    ⊢ LE.le ((fun x1 x2 => HMul.hMul x1 x2) a b) 1
  -/
  convert mul_le_mul' ha hb
  /-
    case h.e'_4
    a b : Ordinal.{u_1}
    ha : LE.le a 1
    hb : LE.le b 1
    ⊢ Eq 1 (HMul.hMul 1 1)
  -/
  exact (mul_one 1).symm
  /-
    🎉 no goals
  -/


theorem principal_mul_of_le_two (ho : o ≤ 2) : Principal (· * ·) o := by
  /-
    o : Ordinal.{u}
    ho : LE.le o 2
    ⊢ Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) o
  -/
  rcases lt_or_eq_of_le ho with (ho | rfl)
    /-
      case inl
      o : Ordinal.{u}
      ho✝ : LE.le o 2
      ho : LT.lt o 2
      ⊢ Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) o
    -/
  · rw [← succ_one, lt_succ_iff] at ho
    /-
      case inl
      o : Ordinal.{u}
      ho✝ : LE.le o 2
      ho : LE.le o 1
      ⊢ Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) o
    -/
    rcases lt_or_eq_of_le ho with (ho | rfl)
      /-
        case inl.inl
        o : Ordinal.{u}
        ho✝¹ : LE.le o 2
        ho✝ : LE.le o 1
        ho : LT.lt o 1
        ⊢ Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) o
      -/
    · rw [lt_one_iff_zero.1 ho]
      /-
        case inl.inl
        o : Ordinal.{u}
        ho✝¹ : LE.le o 2
        ho✝ : LE.le o 1
        ho : LT.lt o 1
        ⊢ Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) 0
      -/
      exact principal_zero
      /-
        🎉 no goals
      -/
      /-
        case inl.inr
        ho✝ : LE.le 1 2
        ho : LE.le 1 1
        ⊢ Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) 1
      -/
    · exact principal_mul_one
      /-
        🎉 no goals
      -/
    /-
      case inr
      ho : LE.le 2 2
      ⊢ Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) 2
    -/
  · exact principal_mul_two
    /-
      🎉 no goals
    -/


theorem principal_add_of_principal_mul (ho : Principal (· * ·) o) (ho₂ : o ≠ 2) :
    Principal (· + ·) o := by
  /-
    o : Ordinal.{u}
    ho : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) o
    ho₂ : Ne o 2
    ⊢ Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) o
  -/
  cases' lt_or_gt_of_ne ho₂ with ho₁ ho₂
    /-
      case inl
      o : Ordinal.{u}
      ho : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) o
      ho₂ : Ne o 2
      ho₁ : LT.lt o 2
      ⊢ Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) o
    -/
  · replace ho₁ : o < succ 1 := by rwa [succ_one]
    /-
      case inl
      o : Ordinal.{u}
      ho : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) o
      ho₂ : Ne o 2
      ho₁ : LT.lt o (Order.succ 1)
      ⊢ Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) o
    -/
    rw [lt_succ_iff] at ho₁
    /-
      case inl
      o : Ordinal.{u}
      ho : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) o
      ho₂ : Ne o 2
      ho₁ : LE.le o 1
      ⊢ Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) o
    -/
    exact principal_add_of_le_one ho₁
    /-
      🎉 no goals
    -/
    /-
      case inr
      o : Ordinal.{u}
      ho : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) o
      ho₂✝ : Ne o 2
      ho₂ : GT.gt o 2
      ⊢ Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) o
    -/
  · refine fun a b hao hbo => lt_of_le_of_lt ?_ (ho (max_lt hao hbo) ho₂)
    /-
      case inr
      o : Ordinal.{u}
      ho : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) o
      ho₂✝ : Ne o 2
      ho₂ : GT.gt o 2
      a b : Ordinal.{u}
      hao : LT.lt a o
      hbo : LT.lt b o
      ⊢ LE.le ((fun x1 x2 => HAdd.hAdd x1 x2) a b) ((fun x1 x2 => HMul.hMul x1 x2) ( …
    -/
    dsimp only
    /-
      case inr
      o : Ordinal.{u}
      ho : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) o
      ho₂✝ : Ne o 2
      ho₂ : GT.gt o 2
      a b : Ordinal.{u}
      hao : LT.lt a o
      hbo : LT.lt b o
      ⊢ LE.le (HAdd.hAdd a b) (HMul.hMul (Max.max a b) 2)
    -/
    rw [← one_add_one_eq_two, mul_add, mul_one]
    /-
      case inr
      o : Ordinal.{u}
      ho : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) o
      ho₂✝ : Ne o 2
      ho₂ : GT.gt o 2
      a b : Ordinal.{u}
      hao : LT.lt a o
      hbo : LT.lt b o
      ⊢ LE.le (HAdd.hAdd a b) (HAdd.hAdd (Max.max a b) (Max.max a b))
    -/
    exact add_le_add (le_max_left a b) (le_max_right a b)
    /-
      🎉 no goals
    -/


theorem isLimit_of_principal_mul (ho₂ : 2 < o) (ho : Principal (· * ·) o) : o.IsLimit :=
  isLimit_of_principal_add ((lt_succ 1).trans (succ_one ▸ ho₂))
    (principal_add_of_principal_mul ho (ne_of_gt ho₂))


@[deprecated (since := "2024-10-16")]
alias principal_mul_isLimit := isLimit_of_principal_mul


theorem principal_mul_iff_mul_left_eq : Principal (· * ·) o ↔ ∀ a, 0 < a → a < o → a * o = o := by
  /-
    o : Ordinal.{u}
    ⊢ Iff (Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) o) (∀ (a : Ordinal.{u} …
  -/
  refine ⟨fun h a ha₀ hao => ?_, fun h a b hao hbo => ?_⟩
    /-
      case refine_1
      o : Ordinal.{u}
      h : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) o
      a : Ordinal.{u}
      ha₀ : LT.lt 0 a
      hao : LT.lt a o
      ⊢ Eq (HMul.hMul a o) o
    -/
  · cases' le_or_gt o 2 with ho ho
      /-
        case refine_1.inl
        o : Ordinal.{u}
        h : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) o
        a : Ordinal.{u}
        ha₀ : LT.lt 0 a
        hao : LT.lt a o
        ho : LE.le o 2
        ⊢ Eq (HMul.hMul a o) o
      -/
    · convert one_mul o
      /-
        case h.e'_2.h.e'_5
        o : Ordinal.{u}
        h : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) o
        a : Ordinal.{u}
        ha₀ : LT.lt 0 a
        hao : LT.lt a o
        ho : LE.le o 2
        ⊢ Eq a 1
      -/
      apply le_antisymm
        /-
          case h.e'_2.h.e'_5.a
          o : Ordinal.{u}
          h : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) o
          a : Ordinal.{u}
          ha₀ : LT.lt 0 a
          hao : LT.lt a o
          ho : LE.le o 2
          ⊢ LE.le a 1
        -/
      · rw [← lt_succ_iff, succ_one]
        /-
          case h.e'_2.h.e'_5.a
          o : Ordinal.{u}
          h : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) o
          a : Ordinal.{u}
          ha₀ : LT.lt 0 a
          hao : LT.lt a o
          ho : LE.le o 2
          ⊢ LT.lt a 2
        -/
        exact hao.trans_le ho
        /-
          🎉 no goals
        -/
        /-
          case h.e'_2.h.e'_5.a
          o : Ordinal.{u}
          h : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) o
          a : Ordinal.{u}
          ha₀ : LT.lt 0 a
          hao : LT.lt a o
          ho : LE.le o 2
          ⊢ LE.le 1 a
        -/
      · rwa [← succ_le_iff, succ_zero] at ha₀
        /-
          🎉 no goals
        -/
      /-
        case refine_1.inr
        o : Ordinal.{u}
        h : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) o
        a : Ordinal.{u}
        ha₀ : LT.lt 0 a
        hao : LT.lt a o
        ho : GT.gt o 2
        ⊢ Eq (HMul.hMul a o) o
      -/
    · exact op_eq_self_of_principal hao (isNormal_mul_right ha₀) h (isLimit_of_principal_mul ho h)
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      o : Ordinal.{u}
      h : ∀ (a : Ordinal.{u}), LT.lt 0 a → LT.lt a o → Eq (HMul.hMul a o) o
      a b : Ordinal.{u}
      hao : LT.lt a o
      hbo : LT.lt b o
      ⊢ LT.lt ((fun x1 x2 => HMul.hMul x1 x2) a b) o
    -/
  · rcases eq_or_ne a 0 with (rfl | ha)
      /-
        case refine_2.inl
        o : Ordinal.{u}
        h : ∀ (a : Ordinal.{u}), LT.lt 0 a → LT.lt a o → Eq (HMul.hMul a o) o
        b : Ordinal.{u}
        hbo : LT.lt b o
        hao : LT.lt 0 o
        ⊢ LT.lt ((fun x1 x2 => HMul.hMul x1 x2) 0 b) o
      -/
    · dsimp only; rwa [zero_mul]
                  /-
                    🎉 no goals
                  -/
    /-
      case refine_2.inr
      o : Ordinal.{u}
      h : ∀ (a : Ordinal.{u}), LT.lt 0 a → LT.lt a o → Eq (HMul.hMul a o) o
      a b : Ordinal.{u}
      hao : LT.lt a o
      hbo : LT.lt b o
      ha : Ne a 0
      ⊢ LT.lt ((fun x1 x2 => HMul.hMul x1 x2) a b) o
    -/
    rw [← Ordinal.pos_iff_ne_zero] at ha
    /-
      case refine_2.inr
      o : Ordinal.{u}
      h : ∀ (a : Ordinal.{u}), LT.lt 0 a → LT.lt a o → Eq (HMul.hMul a o) o
      a b : Ordinal.{u}
      hao : LT.lt a o
      hbo : LT.lt b o
      ha : LT.lt 0 a
      ⊢ LT.lt ((fun x1 x2 => HMul.hMul x1 x2) a b) o
    -/
    rw [← h a ha hao]
    /-
      case refine_2.inr
      o : Ordinal.{u}
      h : ∀ (a : Ordinal.{u}), LT.lt 0 a → LT.lt a o → Eq (HMul.hMul a o) o
      a b : Ordinal.{u}
      hao : LT.lt a o
      hbo : LT.lt b o
      ha : LT.lt 0 a
      ⊢ LT.lt ((fun x1 x2 => HMul.hMul x1 x2) a b) (HMul.hMul a o)
    -/
    exact (isNormal_mul_right ha).strictMono hbo
    /-
      🎉 no goals
    -/


theorem principal_mul_omega0 : Principal (· * ·) ω := fun a b ha hb =>
  match a, b, lt_omega0.1 ha, lt_omega0.1 hb with
  | _, _, ⟨m, rfl⟩, ⟨n, rfl⟩ => by
    /-
      a b : Ordinal.{u_1}
      m n : Nat
      ha : LT.lt (↑m) Ordinal.omega0
      hb : LT.lt (↑n) Ordinal.omega0
      ⊢ LT.lt ((fun x1 x2 => HMul.hMul x1 x2) ↑m ↑n) Ordinal.omega0
    -/
    dsimp only; rw [← natCast_mul]
    /-
      a b : Ordinal.{u_1}
      m n : Nat
      ha : LT.lt (↑m) Ordinal.omega0
      hb : LT.lt (↑n) Ordinal.omega0
      ⊢ LT.lt (↑(HMul.hMul m n)) Ordinal.omega0
    -/
    apply nat_lt_omega0
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-09-30")]
alias principal_mul_omega := principal_mul_omega0


theorem mul_omega0 (a0 : 0 < a) (ha : a < ω) : a * ω = ω :=
  principal_mul_iff_mul_left_eq.1 principal_mul_omega0 a a0 ha


@[deprecated (since := "2024-09-30")]
alias mul_omega := mul_omega0


theorem natCast_mul_omega0 {n : ℕ} (hn : 0 < n) : n * ω = ω :=
  mul_omega0 (mod_cast hn) (nat_lt_omega0 n)


theorem mul_lt_omega0_opow (c0 : 0 < c) (ha : a < ω ^ c) (hb : b < ω) : a * b < ω ^ c := by
  /-
    a b c : Ordinal.{u}
    c0 : LT.lt 0 c
    ha : LT.lt a (HPow.hPow Ordinal.omega0 c)
    hb : LT.lt b Ordinal.omega0
    ⊢ LT.lt (HMul.hMul a b) (HPow.hPow Ordinal.omega0 c)
  -/
  rcases zero_or_succ_or_limit c with (rfl | ⟨c, rfl⟩ | l)
    /-
      case inl
      a b : Ordinal.{u}
      hb : LT.lt b Ordinal.omega0
      c0 : LT.lt 0 0
      ha : LT.lt a (HPow.hPow Ordinal.omega0 0)
      ⊢ LT.lt (HMul.hMul a b) (HPow.hPow Ordinal.omega0 0)
    -/
  · exact (lt_irrefl _).elim c0
    /-
      🎉 no goals
    -/
    /-
      case inr.inl.intro
      a b : Ordinal.{u}
      hb : LT.lt b Ordinal.omega0
      c : Ordinal.{u}
      c0 : LT.lt 0 (Order.succ c)
      ha : LT.lt a (HPow.hPow Ordinal.omega0 (Order.succ c))
      ⊢ LT.lt (HMul.hMul a b) (HPow.hPow Ordinal.omega0 (Order.succ c))
    -/
  · rw [opow_succ] at ha
    obtain ⟨n, hn, an⟩ :=
      ((isNormal_mul_right <| opow_pos _ omega0_pos).limit_lt isLimit_omega0).1 ha
    /-
      case inr.inl.intro.intro.intro
      a b : Ordinal.{u}
      hb : LT.lt b Ordinal.omega0
      c : Ordinal.{u}
      c0 : LT.lt 0 (Order.succ c)
      ha : LT.lt a (HMul.hMul (HPow.hPow Ordinal.omega0 c) Ordinal.omega0)
      n : Ordinal.{u}
      hn : LT.lt n Ordinal.omega0
      an : LT.lt a (HMul.hMul (HPow.hPow Ordinal.omega0 c) n)
      ⊢ LT.lt (HMul.hMul a b) (HPow.hPow Ordinal.omega0 (Order.succ c))
    -/
    apply (mul_le_mul_right' (le_of_lt an) _).trans_lt
    /-
      case inr.inl.intro.intro.intro
      a b : Ordinal.{u}
      hb : LT.lt b Ordinal.omega0
      c : Ordinal.{u}
      c0 : LT.lt 0 (Order.succ c)
      ha : LT.lt a (HMul.hMul (HPow.hPow Ordinal.omega0 c) Ordinal.omega0)
      n : Ordinal.{u}
      hn : LT.lt n Ordinal.omega0
      an : LT.lt a (HMul.hMul (HPow.hPow Ordinal.omega0 c) n)
      ⊢ LT.lt (HMul.hMul (HMul.hMul (HPow.hPow Ordinal.omega0 c) n) b) (HPow.hPow Or …
    -/
    rw [opow_succ, mul_assoc, mul_lt_mul_iff_left (opow_pos _ omega0_pos)]
    /-
      case inr.inl.intro.intro.intro
      a b : Ordinal.{u}
      hb : LT.lt b Ordinal.omega0
      c : Ordinal.{u}
      c0 : LT.lt 0 (Order.succ c)
      ha : LT.lt a (HMul.hMul (HPow.hPow Ordinal.omega0 c) Ordinal.omega0)
      n : Ordinal.{u}
      hn : LT.lt n Ordinal.omega0
      an : LT.lt a (HMul.hMul (HPow.hPow Ordinal.omega0 c) n)
      ⊢ LT.lt (HMul.hMul n b) Ordinal.omega0
    -/
    exact principal_mul_omega0 hn hb
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      a b c : Ordinal.{u}
      c0 : LT.lt 0 c
      ha : LT.lt a (HPow.hPow Ordinal.omega0 c)
      hb : LT.lt b Ordinal.omega0
      l : c.IsLimit
      ⊢ LT.lt (HMul.hMul a b) (HPow.hPow Ordinal.omega0 c)
    -/
  · rcases ((isNormal_opow one_lt_omega0).limit_lt l).1 ha with ⟨x, hx, ax⟩
    /-
      case inr.inr.intro.intro
      a b c : Ordinal.{u}
      c0 : LT.lt 0 c
      ha : LT.lt a (HPow.hPow Ordinal.omega0 c)
      hb : LT.lt b Ordinal.omega0
      l : c.IsLimit
      x : Ordinal.{u}
      hx : LT.lt x c
      ax : LT.lt a (HPow.hPow Ordinal.omega0 x)
      ⊢ LT.lt (HMul.hMul a b) (HPow.hPow Ordinal.omega0 c)
    -/
    refine (mul_le_mul' (le_of_lt ax) (le_of_lt hb)).trans_lt ?_
    /-
      case inr.inr.intro.intro
      a b c : Ordinal.{u}
      c0 : LT.lt 0 c
      ha : LT.lt a (HPow.hPow Ordinal.omega0 c)
      hb : LT.lt b Ordinal.omega0
      l : c.IsLimit
      x : Ordinal.{u}
      hx : LT.lt x c
      ax : LT.lt a (HPow.hPow Ordinal.omega0 x)
      ⊢ LT.lt (HMul.hMul (HPow.hPow Ordinal.omega0 x) Ordinal.omega0) (HPow.hPow Ord …
    -/
    rw [← opow_succ, opow_lt_opow_iff_right one_lt_omega0]
    /-
      case inr.inr.intro.intro
      a b c : Ordinal.{u}
      c0 : LT.lt 0 c
      ha : LT.lt a (HPow.hPow Ordinal.omega0 c)
      hb : LT.lt b Ordinal.omega0
      l : c.IsLimit
      x : Ordinal.{u}
      hx : LT.lt x c
      ax : LT.lt a (HPow.hPow Ordinal.omega0 x)
      ⊢ LT.lt (Order.succ x) c
    -/
    exact l.succ_lt hx
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-09-30")]
alias mul_lt_omega_opow := mul_lt_omega0_opow


theorem mul_omega0_opow_opow (a0 : 0 < a) (h : a < ω ^ ω ^ b) : a * ω ^ ω ^ b = ω ^ ω ^ b := by
  /-
    a b : Ordinal.{u}
    a0 : LT.lt 0 a
    h : LT.lt a (HPow.hPow Ordinal.omega0 (HPow.hPow Ordinal.omega0 b))
    ⊢ Eq (HMul.hMul a (HPow.hPow Ordinal.omega0 (HPow.hPow Ordinal.omega0 b))) (HP …
  -/
  obtain rfl | b0 := eq_or_ne b 0
    /-
      case inl
      a : Ordinal.{u}
      a0 : LT.lt 0 a
      h : LT.lt a (HPow.hPow Ordinal.omega0 (HPow.hPow Ordinal.omega0 0))
      ⊢ Eq (HMul.hMul a (HPow.hPow Ordinal.omega0 (HPow.hPow Ordinal.omega0 0))) (HP …
    -/
  · rw [opow_zero, opow_one] at h ⊢
    /-
      case inl
      a : Ordinal.{u}
      a0 : LT.lt 0 a
      h : LT.lt a Ordinal.omega0
      ⊢ Eq (HMul.hMul a Ordinal.omega0) Ordinal.omega0
    -/
    exact mul_omega0 a0 h
    /-
      🎉 no goals
    -/
    /-
      case inr
      a b : Ordinal.{u}
      a0 : LT.lt 0 a
      h : LT.lt a (HPow.hPow Ordinal.omega0 (HPow.hPow Ordinal.omega0 b))
      b0 : Ne b 0
      ⊢ Eq (HMul.hMul a (HPow.hPow Ordinal.omega0 (HPow.hPow Ordinal.omega0 b))) (HP …
    -/
  · apply le_antisymm
    · obtain ⟨x, xb, ax⟩ :=
        (lt_opow_of_limit omega0_ne_zero (isLimit_opow_left isLimit_omega0 b0)).1 h
      /-
        case inr.a.intro.intro
        a b : Ordinal.{u}
        a0 : LT.lt 0 a
        h : LT.lt a (HPow.hPow Ordinal.omega0 (HPow.hPow Ordinal.omega0 b))
        b0 : Ne b 0
        x : Ordinal.{u}
        xb : LT.lt x (HPow.hPow Ordinal.omega0 b)
        ax : LT.lt a (HPow.hPow Ordinal.omega0 x)
        ⊢ LE.le (HMul.hMul a (HPow.hPow Ordinal.omega0 (HPow.hPow Ordinal.omega0 b)))  …
      -/
      apply (mul_le_mul_right' (le_of_lt ax) _).trans
      /-
        case inr.a.intro.intro
        a b : Ordinal.{u}
        a0 : LT.lt 0 a
        h : LT.lt a (HPow.hPow Ordinal.omega0 (HPow.hPow Ordinal.omega0 b))
        b0 : Ne b 0
        x : Ordinal.{u}
        xb : LT.lt x (HPow.hPow Ordinal.omega0 b)
        ax : LT.lt a (HPow.hPow Ordinal.omega0 x)
        ⊢ LE.le (HMul.hMul (HPow.hPow Ordinal.omega0 x) (HPow.hPow Ordinal.omega0 (HPo …
      -/
      rw [← opow_add, add_omega0_opow xb]
      /-
        🎉 no goals
      -/
      /-
        case inr.a
        a b : Ordinal.{u}
        a0 : LT.lt 0 a
        h : LT.lt a (HPow.hPow Ordinal.omega0 (HPow.hPow Ordinal.omega0 b))
        b0 : Ne b 0
        ⊢ LE.le (HPow.hPow Ordinal.omega0 (HPow.hPow Ordinal.omega0 b)) (HMul.hMul a ( …
      -/
    · conv_lhs => rw [← one_mul (ω ^ _)]
      /-
        case inr.a
        a b : Ordinal.{u}
        a0 : LT.lt 0 a
        h : LT.lt a (HPow.hPow Ordinal.omega0 (HPow.hPow Ordinal.omega0 b))
        b0 : Ne b 0
        ⊢ LE.le (HMul.hMul 1 (HPow.hPow Ordinal.omega0 (HPow.hPow Ordinal.omega0 b)))  …
      -/
      exact mul_le_mul_right' (one_le_iff_pos.2 a0) _
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-09-30")]
alias mul_omega_opow_opow := mul_omega0_opow_opow


theorem principal_mul_omega0_opow_opow (o : Ordinal) : Principal (· * ·) (ω ^ ω ^ o) :=
  principal_mul_iff_mul_left_eq.2 fun _ => mul_omega0_opow_opow


@[deprecated (since := "2024-09-30")]
alias principal_mul_omega_opow_opow := principal_mul_omega0_opow_opow


theorem principal_add_of_principal_mul_opow (hb : 1 < b) (ho : Principal (· * ·) (b ^ o)) :
    Principal (· + ·) o := by
  /-
    b o : Ordinal.{u}
    hb : LT.lt 1 b
    ho : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) (HPow.hPow b o)
    ⊢ Ordinal.Principal (fun x1 x2 => HAdd.hAdd x1 x2) o
  -/
  intro x y hx hy
  /-
    b o : Ordinal.{u}
    hb : LT.lt 1 b
    ho : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) (HPow.hPow b o)
    x y : Ordinal.{u}
    hx : LT.lt x o
    hy : LT.lt y o
    ⊢ LT.lt ((fun x1 x2 => HAdd.hAdd x1 x2) x y) o
  -/
  have := ho ((opow_lt_opow_iff_right hb).2 hx) ((opow_lt_opow_iff_right hb).2 hy)
  /-
    b o : Ordinal.{u}
    hb : LT.lt 1 b
    ho : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) (HPow.hPow b o)
    x y : Ordinal.{u}
    hx : LT.lt x o
    hy : LT.lt y o
    this : LT.lt ((fun x1 x2 => HMul.hMul x1 x2) (HPow.hPow b x) (HPow.hPow b y))  …
    ⊢ LT.lt ((fun x1 x2 => HAdd.hAdd x1 x2) x y) o
  -/
  dsimp only at *
  /-
    b o : Ordinal.{u}
    hb : LT.lt 1 b
    ho : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) (HPow.hPow b o)
    x y : Ordinal.{u}
    hx : LT.lt x o
    hy : LT.lt y o
    this : LT.lt (HMul.hMul (HPow.hPow b x) (HPow.hPow b y)) (HPow.hPow b o)
    ⊢ LT.lt (HAdd.hAdd x y) o
  -/
  rwa [← opow_add, opow_lt_opow_iff_right hb] at this
  /-
    🎉 no goals
  -/


/-- The main characterization theorem for multiplicative principal ordinals. -/
theorem principal_mul_iff_le_two_or_omega0_opow_opow :
    Principal (· * ·) o ↔ o ≤ 2 ∨ o ∈ Set.range (ω ^ ω ^ · : Ordinal → Ordinal) := by
  /-
    o : Ordinal.{u}
    ⊢ Iff (Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) o) (Or (LE.le o 2) (Me …
  -/
  refine ⟨fun ho => ?_, ?_⟩
    /-
      case refine_1
      o : Ordinal.{u}
      ho : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) o
      ⊢ Or (LE.le o 2) (Membership.mem (Set.range fun x => HPow.hPow Ordinal.omega0  …
    -/
  · rcases le_or_lt o 2 with ho₂ | ho₂
      /-
        case refine_1.inl
        o : Ordinal.{u}
        ho : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) o
        ho₂ : LE.le o 2
        ⊢ Or (LE.le o 2) (Membership.mem (Set.range fun x => HPow.hPow Ordinal.omega0  …
      -/
    · exact Or.inl ho₂
      /-
        🎉 no goals
      -/
    · rcases principal_add_iff_zero_or_omega0_opow.1 (principal_add_of_principal_mul ho ho₂.ne')
        with (rfl | ⟨a, rfl⟩)
        /-
          case refine_1.inr.inl
          ho : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) 0
          ho₂ : LT.lt 2 0
          ⊢ Or (LE.le 0 2) (Membership.mem (Set.range fun x => HPow.hPow Ordinal.omega0  …
        -/
      · exact (Ordinal.not_lt_zero 2 ho₂).elim
        /-
          🎉 no goals
        -/
      · rcases principal_add_iff_zero_or_omega0_opow.1
          (principal_add_of_principal_mul_opow one_lt_omega0 ho) with (rfl | ⟨b, rfl⟩)
          /-
            case refine_1.inr.inr.intro.inl
            ho : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) ((fun x => HPow.hPow Ord …
            ho₂ : LT.lt 2 ((fun x => HPow.hPow Ordinal.omega0 x) 0)
            ⊢ Or (LE.le ((fun x => HPow.hPow Ordinal.omega0 x) 0) 2) (Membership.mem (Set. …
          -/
        · simp
          /-
            🎉 no goals
          -/
          /-
            case refine_1.inr.inr.intro.inr.intro
            b : Ordinal.{u}
            ho : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) ((fun x => HPow.hPow Ord …
            ho₂ : LT.lt 2 ((fun x => HPow.hPow Ordinal.omega0 x) ((fun x => HPow.hPow Ordi …
            ⊢ Or (LE.le ((fun x => HPow.hPow Ordinal.omega0 x) ((fun x => HPow.hPow Ordina …
          -/
        · exact Or.inr ⟨b, rfl⟩
          /-
            🎉 no goals
          -/
    /-
      case refine_2
      o : Ordinal.{u}
      ⊢ Or (LE.le o 2) (Membership.mem (Set.range fun x => HPow.hPow Ordinal.omega0  …
    -/
  · rintro (ho₂ | ⟨a, rfl⟩)
      /-
        case refine_2.inl
        o : Ordinal.{u}
        ho₂ : LE.le o 2
        ⊢ Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) o
      -/
    · exact principal_mul_of_le_two ho₂
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr.intro
        a : Ordinal.{u}
        ⊢ Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) ((fun x => HPow.hPow Ordina …
      -/
    · exact principal_mul_omega0_opow_opow a
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-09-30")]
alias principal_mul_iff_le_two_or_omega_opow_opow := principal_mul_iff_le_two_or_omega0_opow_opow


theorem mul_omega0_dvd (a0 : 0 < a) (ha : a < ω) : ∀ {b}, ω ∣ b → a * b = b
                      /-
                        a : Ordinal.{u}
                        a0 : LT.lt 0 a
                        ha : LT.lt a Ordinal.omega0
                        b : Ordinal.{u}
                        ⊢ Eq (HMul.hMul a (HMul.hMul Ordinal.omega0 b)) (HMul.hMul Ordinal.omega0 b)
                      -/
  | _, ⟨b, rfl⟩ => by rw [← mul_assoc, mul_omega0 a0 ha]
                      /-
                        🎉 no goals
                      -/


@[deprecated (since := "2024-09-30")]
alias mul_omega_dvd := mul_omega0_dvd


theorem mul_eq_opow_log_succ (ha : a ≠ 0) (hb : Principal (· * ·) b) (hb₂ : 2 < b) :
    a * b = b ^ succ (log b a) := by
  /-
    a b : Ordinal.{u}
    ha : Ne a 0
    hb : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) b
    hb₂ : LT.lt 2 b
    ⊢ Eq (HMul.hMul a b) (HPow.hPow b (Order.succ (Ordinal.log b a)))
  -/
  apply le_antisymm
    /-
      case a
      a b : Ordinal.{u}
      ha : Ne a 0
      hb : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) b
      hb₂ : LT.lt 2 b
      ⊢ LE.le (HMul.hMul a b) (HPow.hPow b (Order.succ (Ordinal.log b a)))
    -/
  · have hbl := isLimit_of_principal_mul hb₂ hb
    /-
      case a
      a b : Ordinal.{u}
      ha : Ne a 0
      hb : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) b
      hb₂ : LT.lt 2 b
      hbl : b.IsLimit
      ⊢ LE.le (HMul.hMul a b) (HPow.hPow b (Order.succ (Ordinal.log b a)))
    -/
    rw [← (isNormal_mul_right (Ordinal.pos_iff_ne_zero.2 ha)).bsup_eq hbl, bsup_le_iff]
    /-
      case a
      a b : Ordinal.{u}
      ha : Ne a 0
      hb : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) b
      hb₂ : LT.lt 2 b
      hbl : b.IsLimit
      ⊢ ∀ (i : Ordinal.{u}), LT.lt i b → LE.le (HMul.hMul a i) (HPow.hPow b (Order.s …
    -/
    intro c hcb
    /-
      case a
      a b : Ordinal.{u}
      ha : Ne a 0
      hb : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) b
      hb₂ : LT.lt 2 b
      hbl : b.IsLimit
      c : Ordinal.{u}
      hcb : LT.lt c b
      ⊢ LE.le (HMul.hMul a c) (HPow.hPow b (Order.succ (Ordinal.log b a)))
    -/
    have hb₁ : 1 < b := one_lt_two.trans hb₂
    /-
      case a
      a b : Ordinal.{u}
      ha : Ne a 0
      hb : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) b
      hb₂ : LT.lt 2 b
      hbl : b.IsLimit
      c : Ordinal.{u}
      hcb : LT.lt c b
      hb₁ : LT.lt 1 b
      ⊢ LE.le (HMul.hMul a c) (HPow.hPow b (Order.succ (Ordinal.log b a)))
    -/
    have hbo₀ : b ^ log b a ≠ 0 := Ordinal.pos_iff_ne_zero.1 (opow_pos _ (zero_lt_one.trans hb₁))
    /-
      case a
      a b : Ordinal.{u}
      ha : Ne a 0
      hb : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) b
      hb₂ : LT.lt 2 b
      hbl : b.IsLimit
      c : Ordinal.{u}
      hcb : LT.lt c b
      hb₁ : LT.lt 1 b
      hbo₀ : Ne (HPow.hPow b (Ordinal.log b a)) 0
      ⊢ LE.le (HMul.hMul a c) (HPow.hPow b (Order.succ (Ordinal.log b a)))
    -/
    apply (mul_le_mul_right' (le_of_lt (lt_mul_succ_div a hbo₀)) c).trans
    /-
      case a
      a b : Ordinal.{u}
      ha : Ne a 0
      hb : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) b
      hb₂ : LT.lt 2 b
      hbl : b.IsLimit
      c : Ordinal.{u}
      hcb : LT.lt c b
      hb₁ : LT.lt 1 b
      hbo₀ : Ne (HPow.hPow b (Ordinal.log b a)) 0
      ⊢ LE.le (HMul.hMul (HMul.hMul (HPow.hPow b (Ordinal.log b a)) (Order.succ (HDi …
    -/
    rw [mul_assoc, opow_succ]
    /-
      case a
      a b : Ordinal.{u}
      ha : Ne a 0
      hb : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) b
      hb₂ : LT.lt 2 b
      hbl : b.IsLimit
      c : Ordinal.{u}
      hcb : LT.lt c b
      hb₁ : LT.lt 1 b
      hbo₀ : Ne (HPow.hPow b (Ordinal.log b a)) 0
      ⊢ LE.le (HMul.hMul (HPow.hPow b (Ordinal.log b a)) (HMul.hMul (Order.succ (HDi …
    -/
    refine mul_le_mul_left' (hb (hbl.succ_lt ?_) hcb).le _
    /-
      case a
      a b : Ordinal.{u}
      ha : Ne a 0
      hb : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) b
      hb₂ : LT.lt 2 b
      hbl : b.IsLimit
      c : Ordinal.{u}
      hcb : LT.lt c b
      hb₁ : LT.lt 1 b
      hbo₀ : Ne (HPow.hPow b (Ordinal.log b a)) 0
      ⊢ LT.lt (HDiv.hDiv a (HPow.hPow b (Ordinal.log b a))) b
    -/
    rw [div_lt hbo₀, ← opow_succ]
    /-
      case a
      a b : Ordinal.{u}
      ha : Ne a 0
      hb : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) b
      hb₂ : LT.lt 2 b
      hbl : b.IsLimit
      c : Ordinal.{u}
      hcb : LT.lt c b
      hb₁ : LT.lt 1 b
      hbo₀ : Ne (HPow.hPow b (Ordinal.log b a)) 0
      ⊢ LT.lt a (HPow.hPow b (Order.succ (Ordinal.log b a)))
    -/
    exact lt_opow_succ_log_self hb₁ _
    /-
      🎉 no goals
    -/
    /-
      case a
      a b : Ordinal.{u}
      ha : Ne a 0
      hb : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) b
      hb₂ : LT.lt 2 b
      ⊢ LE.le (HPow.hPow b (Order.succ (Ordinal.log b a))) (HMul.hMul a b)
    -/
  · rw [opow_succ]
    /-
      case a
      a b : Ordinal.{u}
      ha : Ne a 0
      hb : Ordinal.Principal (fun x1 x2 => HMul.hMul x1 x2) b
      hb₂ : LT.lt 2 b
      ⊢ LE.le (HMul.hMul (HPow.hPow b (Ordinal.log b a)) b) (HMul.hMul a b)
    -/
    exact mul_le_mul_right' (opow_log_le_self b ha) b
    /-
      🎉 no goals
    -/


theorem principal_opow_omega0 : Principal (· ^ ·) ω := fun a b ha hb =>
  match a, b, lt_omega0.1 ha, lt_omega0.1 hb with
  | _, _, ⟨m, rfl⟩, ⟨n, rfl⟩ => by
    /-
      a b : Ordinal.{u_1}
      m n : Nat
      ha : LT.lt (↑m) Ordinal.omega0
      hb : LT.lt (↑n) Ordinal.omega0
      ⊢ LT.lt ((fun x1 x2 => HPow.hPow x1 x2) ↑m ↑n) Ordinal.omega0
    -/
    simp_rw [← natCast_opow]
    /-
      a b : Ordinal.{u_1}
      m n : Nat
      ha : LT.lt (↑m) Ordinal.omega0
      hb : LT.lt (↑n) Ordinal.omega0
      ⊢ LT.lt (↑(HPow.hPow m n)) Ordinal.omega0
    -/
    apply nat_lt_omega0
    /-
      🎉 no goals
    -/


theorem opow_omega0 (a1 : 1 < a) (h : a < ω) : a ^ ω = ω :=
  ((opow_le_of_limit (one_le_iff_ne_zero.1 <| le_of_lt a1) isLimit_omega0).2 fun _ hb =>
      (principal_opow_omega0 h hb).le).antisymm
  (right_le_opow _ a1)


@[deprecated (since := "2024-09-30")]
alias opow_omega := opow_omega0


theorem natCast_opow_omega0 {n : ℕ} (hn : 1 < n) : n ^ ω = ω :=
  opow_omega0 (mod_cast hn) (nat_lt_omega0 n)


