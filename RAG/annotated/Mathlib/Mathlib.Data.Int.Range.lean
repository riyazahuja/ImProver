/-- List enumerating `[m, n)`. This is the ℤ variant of `List.Ico`. -/
def range (m n : ℤ) : List ℤ :=
  ((List.range (toNat (n - m))) : List ℕ).map fun (r : ℕ) => (m + r : ℤ)


theorem mem_range_iff {m n r : ℤ} : r ∈ range m n ↔ m ≤ r ∧ r < n := by
  /-
    m n r : Int
    ⊢ Iff (Membership.mem (m.range n) r) (And (LE.le m r) (LT.lt r n))
  -/
  simp only [range, List.mem_map, List.mem_range, lt_toNat, lt_sub_iff_add_lt, add_comm]
  exact ⟨fun ⟨a, ha⟩ => ha.2 ▸ ⟨le_add_of_nonneg_right (Int.natCast_nonneg _), ha.1⟩,
    fun h => ⟨toNat (r - m), by simp [toNat_of_nonneg (sub_nonneg.2 h.1), h.2] ⟩⟩


instance decidableLELT (P : Int → Prop) [DecidablePred P] (m n : ℤ) :
    Decidable (∀ r, m ≤ r → r < n → P r) :=
                                                /-
                                                  P : Int → Prop
                                                  inst✝ : DecidablePred P
                                                  m n : Int
                                                  ⊢ Iff (∀ (r : Int), Membership.mem (m.range n) r → P r) (∀ (r : Int), LE.le m  …
                                                -/
  decidable_of_iff (∀ r ∈ range m n, P r) <| by simp only [mem_range_iff, and_imp]
                                                /-
                                                  🎉 no goals
                                                -/


instance decidableLELE (P : Int → Prop) [DecidablePred P] (m n : ℤ) :
    Decidable (∀ r, m ≤ r → r ≤ n → P r) := by
  -- Porting note: The previous code was:
  -- decidable_of_iff (∀ r ∈ range m (n + 1), P r) <| by
  --   simp only [mem_range_iff, and_imp, lt_add_one_iff]
  --
  -- This fails to synthesize an instance
  -- `Decidable (∀ (r : ℤ), r ∈ range m (n + 1) → P r)`
    /-
      P : Int → Prop
      inst✝ : DecidablePred P
      m n : Int
      ⊢ Decidable (∀ (r : Int), LE.le m r → LE.le r n → P r)
    -/
    apply decidable_of_iff (∀ r ∈ range m (n + 1), P r)
    /-
      case h
      P : Int → Prop
      inst✝ : DecidablePred P
      m n : Int
      ⊢ Iff (∀ (r : Int), Membership.mem (m.range (HAdd.hAdd n 1)) r → P r) (∀ (r :  …
    -/
    apply Iff.intro <;> intros h _ _
      /-
        case h.mp
        P : Int → Prop
        inst✝ : DecidablePred P
        m n : Int
        h : ∀ (r : Int), Membership.mem (m.range (HAdd.hAdd n 1)) r → P r
        r✝ : Int
        a✝ : LE.le m r✝
        ⊢ LE.le r✝ n → P r✝
      -/
    · intro _; apply h
      /-
        case h.mp.a
        P : Int → Prop
        inst✝ : DecidablePred P
        m n : Int
        h : ∀ (r : Int), Membership.mem (m.range (HAdd.hAdd n 1)) r → P r
        r✝ : Int
        a✝¹ : LE.le m r✝
        a✝ : LE.le r✝ n
        ⊢ Membership.mem (m.range (HAdd.hAdd n 1)) r✝
      -/
      simp_all only [mem_range_iff, and_imp, and_self, lt_add_one_iff]
      /-
        🎉 no goals
      -/
      /-
        case h.mpr
        P : Int → Prop
        inst✝ : DecidablePred P
        m n : Int
        h : ∀ (r : Int), LE.le m r → LE.le r n → P r
        r✝ : Int
        a✝ : Membership.mem (m.range (HAdd.hAdd n 1)) r✝
        ⊢ P r✝
      -/
    · simp_all only [mem_range_iff, and_imp, lt_add_one_iff]
      /-
        🎉 no goals
      -/


instance decidableLTLT (P : Int → Prop) [DecidablePred P] (m n : ℤ) :
    Decidable (∀ r, m < r → r < n → P r) :=
  Int.decidableLELT P _ _


instance decidableLTLE (P : Int → Prop) [DecidablePred P] (m n : ℤ) :
    Decidable (∀ r, m < r → r ≤ n → P r) :=
  Int.decidableLELE P _ _


