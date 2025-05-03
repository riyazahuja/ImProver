/-- In a linear ordered field, the whole field is order isomorphic to the open interval `(-1, 1)`.
We consider the actual implementation to be a "black box", so it is irreducible.
-/
@[irreducible]
def orderIsoIooNegOneOne (k : Type*) [LinearOrderedField k] : k ≃o Ioo (-1 : k) 1 := by
  /-
    k : Type u_1
    inst✝ : LinearOrderedField k
    ⊢ OrderIso k ↑(Set.Ioo (-1) 1)
  -/
  refine StrictMono.orderIsoOfRightInverse ?_ ?_ (fun x ↦ x / (1 - |↑x|)) ?_
    /-
      case refine_1
      k : Type u_1
      inst✝ : LinearOrderedField k
      ⊢ k → ↑(Set.Ioo (-1) 1)
    -/
  · refine codRestrict (fun x ↦ x / (1 + |x|)) _ fun x ↦ abs_lt.1 ?_
    /-
      case refine_1
      k : Type u_1
      inst✝ : LinearOrderedField k
      x : k
      ⊢ LT.lt (abs ((fun x => HDiv.hDiv x (HAdd.hAdd 1 (abs x))) x)) 1
    -/
    have H : 0 < 1 + |x| := (abs_nonneg x).trans_lt (lt_one_add _)
    calc
      |x / (1 + |x|)| = |x| / (1 + |x|) := by rw [abs_div, abs_of_pos H]
      _ < 1 := (div_lt_one H).2 (lt_one_add _)
    /-
      case refine_2
      k : Type u_1
      inst✝ : LinearOrderedField k
      ⊢ StrictMono (Set.codRestrict (fun x => HDiv.hDiv x (HAdd.hAdd 1 (abs x))) (Se …
    -/
  · refine (strictMono_of_odd_strictMonoOn_nonneg ?_ ?_).codRestrict _
      /-
        case refine_2.refine_1
        k : Type u_1
        inst✝ : LinearOrderedField k
        ⊢ ∀ (x : k), Eq (HDiv.hDiv (Neg.neg x) (HAdd.hAdd 1 (abs (Neg.neg x)))) (Neg.n …
      -/
    · intro x
      /-
        case refine_2.refine_1
        k : Type u_1
        inst✝ : LinearOrderedField k
        x : k
        ⊢ Eq (HDiv.hDiv (Neg.neg x) (HAdd.hAdd 1 (abs (Neg.neg x)))) (Neg.neg (HDiv.hD …
      -/
      simp only [abs_neg, neg_div]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        k : Type u_1
        inst✝ : LinearOrderedField k
        ⊢ StrictMonoOn (fun x => HDiv.hDiv x (HAdd.hAdd 1 (abs x))) (Set.Ici 0)
      -/
    · rintro x (hx : 0 ≤ x) y (hy : 0 ≤ y) hxy
      simp [abs_of_nonneg, mul_add, mul_comm x y, div_lt_div_iff₀, hx.trans_lt (lt_one_add _),
        hy.trans_lt (lt_one_add _), *]
    /-
      case refine_3
      k : Type u_1
      inst✝ : LinearOrderedField k
      ⊢ Function.RightInverse (fun x => HDiv.hDiv (↑x) (HSub.hSub 1 (abs ↑x))) (Set. …
    -/
  · refine fun x ↦ Subtype.ext ?_
    /-
      case refine_3
      k : Type u_1
      inst✝ : LinearOrderedField k
      x : ↑(Set.Ioo (-1) 1)
      ⊢ Eq ↑(Set.codRestrict (fun x => HDiv.hDiv x (HAdd.hAdd 1 (abs x))) (Set.Ioo ( …
    -/
    have : 0 < 1 - |(x : k)| := sub_pos.2 (abs_lt.2 x.2)
    /-
      case refine_3
      k : Type u_1
      inst✝ : LinearOrderedField k
      x : ↑(Set.Ioo (-1) 1)
      this : LT.lt 0 (HSub.hSub 1 (abs ↑x))
      ⊢ Eq ↑(Set.codRestrict (fun x => HDiv.hDiv x (HAdd.hAdd 1 (abs x))) (Set.Ioo ( …
    -/
    field_simp [abs_div, this.ne', abs_of_pos this]
    /-
      🎉 no goals
    -/

