/-- Center of mass of a finite collection of points with prescribed weights.
Note that we require neither `0 ≤ w i` nor `∑ w = 1`. -/
def Finset.centerMass (t : Finset ι) (w : ι → R) (z : ι → E) : E :=
  (∑ i ∈ t, w i)⁻¹ • ∑ i ∈ t, w i • z i


theorem Finset.centerMass_empty : (∅ : Finset ι).centerMass w z = 0 := by
  /-
    R : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    w : ι → R
    z : ι → E
    ⊢ Eq (EmptyCollection.emptyCollection.centerMass w z) 0
  -/
  simp only [centerMass, sum_empty, smul_zero]
  /-
    🎉 no goals
  -/


open scoped Classical in
theorem Finset.centerMass_pair (hne : i ≠ j) :
    ({i, j} : Finset ι).centerMass w z = (w i / (w i + w j)) • z i + (w j / (w i + w j)) • z j := by
  /-
    R : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    i j : ι
    w : ι → R
    z : ι → E
    hne : Ne i j
    ⊢ Eq ((Insert.insert i (Singleton.singleton j)).centerMass w z) (HAdd.hAdd (HS …
  -/
  simp only [centerMass, sum_pair hne]
  /-
    R : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    i j : ι
    w : ι → R
    z : ι → E
    hne : Ne i j
    ⊢ Eq (HSMul.hSMul (Inv.inv (HAdd.hAdd (w i) (w j))) (HAdd.hAdd (HSMul.hSMul (w …
  -/
  module
  /-
    🎉 no goals
  -/


open scoped Classical in
theorem Finset.centerMass_insert (ha : i ∉ t) (hw : ∑ j ∈ t, w j ≠ 0) :
    (insert i t).centerMass w z =
      (w i / (w i + ∑ j ∈ t, w j)) • z i +
        ((∑ j ∈ t, w j) / (w i + ∑ j ∈ t, w j)) • t.centerMass w z := by
  /-
    R : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    i : ι
    t : Finset ι
    w : ι → R
    z : ι → E
    ha : Not (Membership.mem t i)
    hw : Ne (t.sum fun j => w j) 0
    ⊢ Eq ((Insert.insert i t).centerMass w z) (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv ( …
  -/
  simp only [centerMass, sum_insert ha, smul_add, (mul_smul _ _ _).symm, ← div_eq_inv_mul]
  /-
    R : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    i : ι
    t : Finset ι
    w : ι → R
    z : ι → E
    ha : Not (Membership.mem t i)
    hw : Ne (t.sum fun j => w j) 0
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv (w i) (HAdd.hAdd (w i) (t.sum fun i => …
  -/
  congr 2
  /-
    case e_a.e_a
    R : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    i : ι
    t : Finset ι
    w : ι → R
    z : ι → E
    ha : Not (Membership.mem t i)
    hw : Ne (t.sum fun j => w j) 0
    ⊢ Eq (Inv.inv (HAdd.hAdd (w i) (t.sum fun i => w i))) (HMul.hMul (HDiv.hDiv (t …
  -/
  rw [div_mul_eq_mul_div, mul_inv_cancel₀ hw, one_div]
  /-
    🎉 no goals
  -/


theorem Finset.centerMass_singleton (hw : w i ≠ 0) : ({i} : Finset ι).centerMass w z = z i := by
  /-
    R : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    i : ι
    w : ι → R
    z : ι → E
    hw : Ne (w i) 0
    ⊢ Eq ((Singleton.singleton i).centerMass w z) (z i)
  -/
  rw [centerMass, sum_singleton, sum_singleton]
  /-
    R : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    i : ι
    w : ι → R
    z : ι → E
    hw : Ne (w i) 0
    ⊢ Eq (HSMul.hSMul (Inv.inv (w i)) (HSMul.hSMul (w i) (z i))) (z i)
  -/
  match_scalars
  /-
    R : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    i : ι
    w : ι → R
    z : ι → E
    hw : Ne (w i) 0
    ⊢ Eq (HMul.hMul (Inv.inv (w i)) (HMul.hMul (w i) 1)) 1
  -/
  field_simp
  /-
    🎉 no goals
  -/


@[simp] lemma Finset.centerMass_neg_left : t.centerMass (-w) z = t.centerMass w z := by
  /-
    R : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    t : Finset ι
    w : ι → R
    z : ι → E
    ⊢ Eq (t.centerMass (Neg.neg w) z) (t.centerMass w z)
  -/
  simp [centerMass, inv_neg]
  /-
    🎉 no goals
  -/


lemma Finset.centerMass_smul_left {c : R'} [Module R' R] [Module R' E] [SMulCommClass R' R R]
    [IsScalarTower R' R R] [SMulCommClass R R' E] [IsScalarTower R' R E] (hc : c ≠ 0) :
    t.centerMass (c • w) z = t.centerMass w z := by
  /-
    R : Type u_1
    R' : Type u_2
    E : Type u_3
    ι : Type u_5
    inst✝⁹ : LinearOrderedField R
    inst✝⁸ : LinearOrderedField R'
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module R E
    t : Finset ι
    w : ι → R
    z : ι → E
    c : R'
    inst✝⁵ : Module R' R
    inst✝⁴ : Module R' E
    inst✝³ : SMulCommClass R' R R
    inst✝² : IsScalarTower R' R R
    inst✝¹ : SMulCommClass R R' E
    inst✝ : IsScalarTower R' R E
    hc : Ne c 0
    ⊢ Eq (t.centerMass (HSMul.hSMul c w) z) (t.centerMass w z)
  -/
  simp [centerMass, -smul_assoc, smul_assoc c, ← smul_sum, smul_inv₀, smul_smul_smul_comm, hc]
  /-
    🎉 no goals
  -/


theorem Finset.centerMass_eq_of_sum_1 (hw : ∑ i ∈ t, w i = 1) :
    t.centerMass w z = ∑ i ∈ t, w i • z i := by
  /-
    R : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    t : Finset ι
    w : ι → R
    z : ι → E
    hw : Eq (t.sum fun i => w i) 1
    ⊢ Eq (t.centerMass w z) (t.sum fun i => HSMul.hSMul (w i) (z i))
  -/
  simp only [Finset.centerMass, hw, inv_one, one_smul]
  /-
    🎉 no goals
  -/


theorem Finset.centerMass_smul : (t.centerMass w fun i => c • z i) = c • t.centerMass w z := by
  /-
    R : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    c : R
    t : Finset ι
    w : ι → R
    z : ι → E
    ⊢ Eq (t.centerMass w fun i => HSMul.hSMul c (z i)) (HSMul.hSMul c (t.centerMas …
  -/
  simp only [Finset.centerMass, Finset.smul_sum, (mul_smul _ _ _).symm, mul_comm c, mul_assoc]
  /-
    🎉 no goals
  -/


/-- A convex combination of two centers of mass is a center of mass as well. This version
deals with two different index types. -/
theorem Finset.centerMass_segment' (s : Finset ι) (t : Finset ι') (ws : ι → R) (zs : ι → E)
    (wt : ι' → R) (zt : ι' → E) (hws : ∑ i ∈ s, ws i = 1) (hwt : ∑ i ∈ t, wt i = 1) (a b : R)
    (hab : a + b = 1) : a • s.centerMass ws zs + b • t.centerMass wt zt = (s.disjSum t).centerMass
    (Sum.elim (fun i => a * ws i) fun j => b * wt j) (Sum.elim zs zt) := by
  rw [s.centerMass_eq_of_sum_1 _ hws, t.centerMass_eq_of_sum_1 _ hwt, smul_sum, smul_sum, ←
    Finset.sum_sum_elim, Finset.centerMass_eq_of_sum_1]
    /-
      R : Type u_1
      E : Type u_3
      ι : Type u_5
      ι' : Type u_6
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      s : Finset ι
      t : Finset ι'
      ws : ι → R
      zs : ι → E
      wt : ι' → R
      zt : ι' → E
      hws : Eq (s.sum fun i => ws i) 1
      hwt : Eq (t.sum fun i => wt i) 1
      a b : R
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ Eq ((s.disjSum t).sum fun x => Sum.elim (fun x => HSMul.hSMul a (HSMul.hSMul …
    -/
                      /-
                        🎉 no goals
                      -/
  · congr with ⟨⟩ <;> simp only [Sum.elim_inl, Sum.elim_inr, mul_smul]
                      /-
                        🎉 no goals
                      -/
    /-
      case hw
      R : Type u_1
      E : Type u_3
      ι : Type u_5
      ι' : Type u_6
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      s : Finset ι
      t : Finset ι'
      ws : ι → R
      zs : ι → E
      wt : ι' → R
      zt : ι' → E
      hws : Eq (s.sum fun i => ws i) 1
      hwt : Eq (t.sum fun i => wt i) 1
      a b : R
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ Eq ((s.disjSum t).sum fun i => Sum.elim (fun i => HMul.hMul a (ws i)) (fun j …
    -/
  · rw [sum_sum_elim, ← mul_sum, ← mul_sum, hws, hwt, mul_one, mul_one, hab]
    /-
      🎉 no goals
    -/


/-- A convex combination of two centers of mass is a center of mass as well. This version
works if two centers of mass share the set of original points. -/
theorem Finset.centerMass_segment (s : Finset ι) (w₁ w₂ : ι → R) (z : ι → E)
    (hw₁ : ∑ i ∈ s, w₁ i = 1) (hw₂ : ∑ i ∈ s, w₂ i = 1) (a b : R) (hab : a + b = 1) :
    a • s.centerMass w₁ z + b • s.centerMass w₂ z =
    s.centerMass (fun i => a * w₁ i + b * w₂ i) z := by
  have hw : (∑ i ∈ s, (a * w₁ i + b * w₂ i)) = 1 := by
    simp only [← mul_sum, sum_add_distrib, mul_one, *]
  simp only [Finset.centerMass_eq_of_sum_1, Finset.centerMass_eq_of_sum_1 _ _ hw,
    smul_sum, sum_add_distrib, add_smul, mul_smul, *]


open scoped Classical in
theorem Finset.centerMass_ite_eq (hi : i ∈ t) :
    t.centerMass (fun j => if i = j then (1 : R) else 0) z = z i := by
  /-
    R : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    i : ι
    t : Finset ι
    z : ι → E
    hi : Membership.mem t i
    ⊢ Eq (t.centerMass (fun j => ite (Eq i j) 1 0) z) (z i)
  -/
  rw [Finset.centerMass_eq_of_sum_1]
    /-
      R : Type u_1
      E : Type u_3
      ι : Type u_5
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      i : ι
      t : Finset ι
      z : ι → E
      hi : Membership.mem t i
      ⊢ Eq (t.sum fun i_1 => HSMul.hSMul (ite (Eq i i_1) 1 0) (z i_1)) (z i)
    -/
  · trans ∑ j ∈ t, if i = j then z i else 0
      /-
        R : Type u_1
        E : Type u_3
        ι : Type u_5
        inst✝² : LinearOrderedField R
        inst✝¹ : AddCommGroup E
        inst✝ : Module R E
        i : ι
        t : Finset ι
        z : ι → E
        hi : Membership.mem t i
        ⊢ Eq (t.sum fun i_1 => HSMul.hSMul (ite (Eq i i_1) 1 0) (z i_1)) (t.sum fun j  …
      -/
    · congr with i
      /-
        case e_f.h
        R : Type u_1
        E : Type u_3
        ι : Type u_5
        inst✝² : LinearOrderedField R
        inst✝¹ : AddCommGroup E
        inst✝ : Module R E
        i✝ : ι
        t : Finset ι
        z : ι → E
        hi : Membership.mem t i✝
        i : ι
        ⊢ Eq (HSMul.hSMul (ite (Eq i✝ i) 1 0) (z i)) (ite (Eq i✝ i) (z i✝) 0)
      -/
      split_ifs with h
      /-
        case pos
        R : Type u_1
        E : Type u_3
        ι : Type u_5
        inst✝² : LinearOrderedField R
        inst✝¹ : AddCommGroup E
        inst✝ : Module R E
        i✝ : ι
        t : Finset ι
        z : ι → E
        hi : Membership.mem t i✝
        i : ι
        h : Eq i✝ i
        ⊢ Eq (HSMul.hSMul 1 (z i)) (z i✝)
      -/
      exacts [h ▸ one_smul _ _, zero_smul _ _]
      /-
        🎉 no goals
      -/
      /-
        R : Type u_1
        E : Type u_3
        ι : Type u_5
        inst✝² : LinearOrderedField R
        inst✝¹ : AddCommGroup E
        inst✝ : Module R E
        i : ι
        t : Finset ι
        z : ι → E
        hi : Membership.mem t i
        ⊢ Eq (t.sum fun j => ite (Eq i j) (z i) 0) (z i)
      -/
    · rw [sum_ite_eq, if_pos hi]
      /-
        🎉 no goals
      -/
    /-
      case hw
      R : Type u_1
      E : Type u_3
      ι : Type u_5
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      i : ι
      t : Finset ι
      z : ι → E
      hi : Membership.mem t i
      ⊢ Eq (t.sum fun i_1 => ite (Eq i i_1) 1 0) 1
    -/
  · rw [sum_ite_eq, if_pos hi]
    /-
      🎉 no goals
    -/


theorem Finset.centerMass_subset {t' : Finset ι} (ht : t ⊆ t') (h : ∀ i ∈ t', i ∉ t → w i = 0) :
    t.centerMass w z = t'.centerMass w z := by
  /-
    R : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    t : Finset ι
    w : ι → R
    z : ι → E
    t' : Finset ι
    ht : HasSubset.Subset t t'
    h : ∀ (i : ι), Membership.mem t' i → Not (Membership.mem t i) → Eq (w i) 0
    ⊢ Eq (t.centerMass w z) (t'.centerMass w z)
  -/
  rw [centerMass, sum_subset ht h, smul_sum, centerMass, smul_sum]
  /-
    R : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    t : Finset ι
    w : ι → R
    z : ι → E
    t' : Finset ι
    ht : HasSubset.Subset t t'
    h : ∀ (i : ι), Membership.mem t' i → Not (Membership.mem t i) → Eq (w i) 0
    ⊢ Eq (t.sum fun x => HSMul.hSMul (Inv.inv (t'.sum fun x => w x)) (HSMul.hSMul  …
  -/
  apply sum_subset ht
  /-
    R : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    t : Finset ι
    w : ι → R
    z : ι → E
    t' : Finset ι
    ht : HasSubset.Subset t t'
    h : ∀ (i : ι), Membership.mem t' i → Not (Membership.mem t i) → Eq (w i) 0
    ⊢ ∀ (x : ι), Membership.mem t' x → Not (Membership.mem t x) → Eq (HSMul.hSMul  …
  -/
  intro i hit' hit
  /-
    R : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    t : Finset ι
    w : ι → R
    z : ι → E
    t' : Finset ι
    ht : HasSubset.Subset t t'
    h : ∀ (i : ι), Membership.mem t' i → Not (Membership.mem t i) → Eq (w i) 0
    i : ι
    hit' : Membership.mem t' i
    hit : Not (Membership.mem t i)
    ⊢ Eq (HSMul.hSMul (Inv.inv (t'.sum fun x => w x)) (HSMul.hSMul (w i) (z i))) 0
  -/
  rw [h i hit' hit, zero_smul, smul_zero]
  /-
    🎉 no goals
  -/


theorem Finset.centerMass_filter_ne_zero : {i ∈ t | w i ≠ 0}.centerMass w z = t.centerMass w z :=
  Finset.centerMass_subset z (filter_subset _ _) fun i hit hit' => by
    /-
      R : Type u_1
      E : Type u_3
      ι : Type u_5
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      t : Finset ι
      w : ι → R
      z : ι → E
      i : ι
      hit : Membership.mem t i
      hit' : Not (Membership.mem (Finset.filter (fun i => Ne (w i) 0) t) i)
      ⊢ Eq (w i) 0
    -/
    simpa only [hit, mem_filter, true_and, Ne, Classical.not_not] using hit'
    /-
      🎉 no goals
    -/


theorem centerMass_le_sup {s : Finset ι} {f : ι → α} {w : ι → R} (hw₀ : ∀ i ∈ s, 0 ≤ w i)
    (hw₁ : 0 < ∑ i ∈ s, w i) :
                                                          /-
                                                            R : Type u_1
                                                            R' : Type u_2
                                                            E : Type u_3
                                                            F : Type u_4
                                                            ι : Type u_5
                                                            ι' : Type u_6
                                                            α : Type u_7
                                                            inst✝⁸ : LinearOrderedField R
                                                            inst✝⁷ : LinearOrderedField R'
                                                            inst✝⁶ : AddCommGroup E
                                                            inst✝⁵ : AddCommGroup F
                                                            inst✝⁴ : LinearOrderedAddCommGroup α
                                                            inst✝³ : Module R E
                                                            inst✝² : Module R F
                                                            inst✝¹ : Module R α
                                                            inst✝ : OrderedSMul R α
                                                            s✝ : Set E
                                                            i j : ι
                                                            c : R
                                                            t : Finset ι
                                                            w✝ : ι → R
                                                            z : ι → E
                                                            s : Finset ι
                                                            f : ι → α
                                                            w : ι → R
                                                            hw₀ : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
                                                            hw₁ : LT.lt 0 (s.sum fun i => w i)
                                                            ⊢ Ne s EmptyCollection.emptyCollection
                                                          -/
    s.centerMass w f ≤ s.sup' (nonempty_of_ne_empty <| by rintro rfl; simp at hw₁) f := by
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
  /-
    R : Type u_1
    ι : Type u_5
    α : Type u_7
    inst✝³ : LinearOrderedField R
    inst✝² : LinearOrderedAddCommGroup α
    inst✝¹ : Module R α
    inst✝ : OrderedSMul R α
    s : Finset ι
    f : ι → α
    w : ι → R
    hw₀ : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
    hw₁ : LT.lt 0 (s.sum fun i => w i)
    ⊢ LE.le (s.centerMass w f) (s.sup' ⋯ f)
  -/
  rw [centerMass, inv_smul_le_iff_of_pos hw₁, sum_smul]
  /-
    R : Type u_1
    ι : Type u_5
    α : Type u_7
    inst✝³ : LinearOrderedField R
    inst✝² : LinearOrderedAddCommGroup α
    inst✝¹ : Module R α
    inst✝ : OrderedSMul R α
    s : Finset ι
    f : ι → α
    w : ι → R
    hw₀ : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
    hw₁ : LT.lt 0 (s.sum fun i => w i)
    ⊢ LE.le (s.sum fun i => HSMul.hSMul (w i) (f i)) (s.sum fun i => HSMul.hSMul ( …
  -/
  exact sum_le_sum fun i hi => smul_le_smul_of_nonneg_left (le_sup' _ hi) <| hw₀ i hi
  /-
    🎉 no goals
  -/


theorem inf_le_centerMass {s : Finset ι} {f : ι → α} {w : ι → R} (hw₀ : ∀ i ∈ s, 0 ≤ w i)
    (hw₁ : 0 < ∑ i ∈ s, w i) :
                                       /-
                                         R : Type u_1
                                         R' : Type u_2
                                         E : Type u_3
                                         F : Type u_4
                                         ι : Type u_5
                                         ι' : Type u_6
                                         α : Type u_7
                                         inst✝⁸ : LinearOrderedField R
                                         inst✝⁷ : LinearOrderedField R'
                                         inst✝⁶ : AddCommGroup E
                                         inst✝⁵ : AddCommGroup F
                                         inst✝⁴ : LinearOrderedAddCommGroup α
                                         inst✝³ : Module R E
                                         inst✝² : Module R F
                                         inst✝¹ : Module R α
                                         inst✝ : OrderedSMul R α
                                         s✝ : Set E
                                         i j : ι
                                         c : R
                                         t : Finset ι
                                         w✝ : ι → R
                                         z : ι → E
                                         s : Finset ι
                                         f : ι → α
                                         w : ι → R
                                         hw₀ : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
                                         hw₁ : LT.lt 0 (s.sum fun i => w i)
                                         ⊢ Ne s EmptyCollection.emptyCollection
                                       -/
    s.inf' (nonempty_of_ne_empty <| by rintro rfl; simp at hw₁) f ≤ s.centerMass w f :=
                                                   /-
                                                     🎉 no goals
                                                   -/
  @centerMass_le_sup R _ αᵒᵈ _ _ _ _ _ _ _ hw₀ hw₁


lemma Finset.centerMass_of_sum_add_sum_eq_zero {s t : Finset ι}
    (hw : ∑ i ∈ s, w i + ∑ i ∈ t, w i = 0) (hz : ∑ i ∈ s, w i • z i + ∑ i ∈ t, w i • z i = 0) :
    s.centerMass w z = t.centerMass w z := by
  /-
    R : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    w : ι → R
    z : ι → E
    s t : Finset ι
    hw : Eq (HAdd.hAdd (s.sum fun i => w i) (t.sum fun i => w i)) 0
    hz : Eq (HAdd.hAdd (s.sum fun i => HSMul.hSMul (w i) (z i)) (t.sum fun i => HS …
    ⊢ Eq (s.centerMass w z) (t.centerMass w z)
  -/
  simp [centerMass, eq_neg_of_add_eq_zero_right hw, eq_neg_of_add_eq_zero_left hz, ← neg_inv]
  /-
    🎉 no goals
  -/


/-- The center of mass of a finite subset of a convex set belongs to the set
provided that all weights are non-negative, and the total weight is positive. -/
theorem Convex.centerMass_mem (hs : Convex R s) :
    (∀ i ∈ t, 0 ≤ w i) → (0 < ∑ i ∈ t, w i) → (∀ i ∈ t, z i ∈ s) → t.centerMass w z ∈ s := by
  classical
  induction' t using Finset.induction with i t hi ht
  · simp [lt_irrefl]
  intro h₀ hpos hmem
  have zi : z i ∈ s := hmem _ (mem_insert_self _ _)
  have hs₀ : ∀ j ∈ t, 0 ≤ w j := fun j hj => h₀ j <| mem_insert_of_mem hj
  rw [sum_insert hi] at hpos
  by_cases hsum_t : ∑ j ∈ t, w j = 0
  · have ws : ∀ j ∈ t, w j = 0 := (sum_eq_zero_iff_of_nonneg hs₀).1 hsum_t
    have wz : ∑ j ∈ t, w j • z j = 0 := sum_eq_zero fun i hi => by simp [ws i hi]
    simp only [centerMass, sum_insert hi, wz, hsum_t, add_zero]
    simp only [hsum_t, add_zero] at hpos
    rw [← mul_smul, inv_mul_cancel₀ (ne_of_gt hpos), one_smul]
    exact zi
  · rw [Finset.centerMass_insert _ _ _ hi hsum_t]
    refine convex_iff_div.1 hs zi (ht hs₀ ?_ ?_) ?_ (sum_nonneg hs₀) hpos
    · exact lt_of_le_of_ne (sum_nonneg hs₀) (Ne.symm hsum_t)
    · intro j hj
      exact hmem j (mem_insert_of_mem hj)
    · exact h₀ _ (mem_insert_self _ _)


theorem Convex.sum_mem (hs : Convex R s) (h₀ : ∀ i ∈ t, 0 ≤ w i) (h₁ : ∑ i ∈ t, w i = 1)
    (hz : ∀ i ∈ t, z i ∈ s) : (∑ i ∈ t, w i • z i) ∈ s := by
  simpa only [h₁, centerMass, inv_one, one_smul] using
    hs.centerMass_mem h₀ (h₁.symm ▸ zero_lt_one) hz


/-- A version of `Convex.sum_mem` for `finsum`s. If `s` is a convex set, `w : ι → R` is a family of
nonnegative weights with sum one and `z : ι → E` is a family of elements of a module over `R` such
that `z i ∈ s` whenever `w i ≠ 0`, then the sum `∑ᶠ i, w i • z i` belongs to `s`. See also
`PartitionOfUnity.finsum_smul_mem_convex`. -/
theorem Convex.finsum_mem {ι : Sort*} {w : ι → R} {z : ι → E} {s : Set E} (hs : Convex R s)
    (h₀ : ∀ i, 0 ≤ w i) (h₁ : ∑ᶠ i, w i = 1) (hz : ∀ i, w i ≠ 0 → z i ∈ s) :
    (∑ᶠ i, w i • z i) ∈ s := by
  have hfin_w : (support (w ∘ PLift.down)).Finite := by
    by_contra H
    rw [finsum, dif_neg H] at h₁
    exact zero_ne_one h₁
  have hsub : support ((fun i => w i • z i) ∘ PLift.down) ⊆ hfin_w.toFinset :=
    (support_smul_subset_left _ _).trans hfin_w.coe_toFinset.ge
  /-
    R : Type u_1
    E : Type u_3
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    ι : Sort u_8
    w : ι → R
    z : ι → E
    s : Set E
    hs : Convex R s
    h₀ : ∀ (i : ι), LE.le 0 (w i)
    h₁ : Eq (finsum fun i => w i) 1
    hz : ∀ (i : ι), Ne (w i) 0 → Membership.mem s (z i)
    hfin_w : (Function.support (Function.comp w PLift.down)).Finite
    hsub : HasSubset.Subset (Function.support (Function.comp (fun i => HSMul.hSMul …
    ⊢ Membership.mem s (finsum fun i => HSMul.hSMul (w i) (z i))
  -/
  rw [finsum_eq_sum_plift_of_support_subset hsub]
  /-
    R : Type u_1
    E : Type u_3
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    ι : Sort u_8
    w : ι → R
    z : ι → E
    s : Set E
    hs : Convex R s
    h₀ : ∀ (i : ι), LE.le 0 (w i)
    h₁ : Eq (finsum fun i => w i) 1
    hz : ∀ (i : ι), Ne (w i) 0 → Membership.mem s (z i)
    hfin_w : (Function.support (Function.comp w PLift.down)).Finite
    hsub : HasSubset.Subset (Function.support (Function.comp (fun i => HSMul.hSMul …
    ⊢ Membership.mem s (hfin_w.toFinset.sum fun i => HSMul.hSMul (w i.down) (z i.d …
  -/
  refine hs.sum_mem (fun _ _ => h₀ _) ?_ fun i hi => hz _ ?_
    /-
      case refine_1
      R : Type u_1
      E : Type u_3
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      ι : Sort u_8
      w : ι → R
      z : ι → E
      s : Set E
      hs : Convex R s
      h₀ : ∀ (i : ι), LE.le 0 (w i)
      h₁ : Eq (finsum fun i => w i) 1
      hz : ∀ (i : ι), Ne (w i) 0 → Membership.mem s (z i)
      hfin_w : (Function.support (Function.comp w PLift.down)).Finite
      hsub : HasSubset.Subset (Function.support (Function.comp (fun i => HSMul.hSMul …
      ⊢ Eq (hfin_w.toFinset.sum fun i => w i.down) 1
    -/
  · rwa [finsum, dif_pos hfin_w] at h₁
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      E : Type u_3
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      ι : Sort u_8
      w : ι → R
      z : ι → E
      s : Set E
      hs : Convex R s
      h₀ : ∀ (i : ι), LE.le 0 (w i)
      h₁ : Eq (finsum fun i => w i) 1
      hz : ∀ (i : ι), Ne (w i) 0 → Membership.mem s (z i)
      hfin_w : (Function.support (Function.comp w PLift.down)).Finite
      hsub : HasSubset.Subset (Function.support (Function.comp (fun i => HSMul.hSMul …
      i : PLift ι
      hi : Membership.mem hfin_w.toFinset i
      ⊢ Ne (w i.down) 0
    -/
  · rwa [hfin_w.mem_toFinset] at hi
    /-
      🎉 no goals
    -/


theorem convex_iff_sum_mem : Convex R s ↔ ∀ (t : Finset E) (w : E → R),
    (∀ i ∈ t, 0 ≤ w i) → ∑ i ∈ t, w i = 1 → (∀ x ∈ t, x ∈ s) → (∑ x ∈ t, w x • x) ∈ s := by
  classical
  refine ⟨fun hs t w hw₀ hw₁ hts => hs.sum_mem hw₀ hw₁ hts, ?_⟩
  intro h x hx y hy a b ha hb hab
  by_cases h_cases : x = y
  · rw [h_cases, ← add_smul, hab, one_smul]
    exact hy
  · convert h {x, y} (fun z => if z = y then b else a) _ _ _
    -- Porting note: Original proof had 2 `simp_intro i hi`
    · simp only [sum_pair h_cases, if_neg h_cases, if_pos trivial]
    · intro i _
      simp only
      split_ifs <;> assumption
    · simp only [sum_pair h_cases, if_neg h_cases, if_pos trivial, hab]
    · intro i hi
      simp only [Finset.mem_singleton, Finset.mem_insert] at hi
      cases hi <;> subst i <;> assumption


theorem Finset.centerMass_mem_convexHull (t : Finset ι) {w : ι → R} (hw₀ : ∀ i ∈ t, 0 ≤ w i)
    (hws : 0 < ∑ i ∈ t, w i) {z : ι → E} (hz : ∀ i ∈ t, z i ∈ s) :
    t.centerMass w z ∈ convexHull R s :=
  (convex_convexHull R s).centerMass_mem hw₀ hws fun i hi => subset_convexHull R s <| hz i hi


/-- A version of `Finset.centerMass_mem_convexHull` for when the weights are nonpositive. -/
lemma Finset.centerMass_mem_convexHull_of_nonpos (t : Finset ι) (hw₀ : ∀ i ∈ t, w i ≤ 0)
    (hws : ∑ i ∈ t, w i < 0) (hz : ∀ i ∈ t, z i ∈ s) : t.centerMass w z ∈ convexHull R s := by
  /-
    R : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    s : Set E
    w : ι → R
    z : ι → E
    t : Finset ι
    hw₀ : ∀ (i : ι), Membership.mem t i → LE.le (w i) 0
    hws : LT.lt (t.sum fun i => w i) 0
    hz : ∀ (i : ι), Membership.mem t i → Membership.mem s (z i)
    ⊢ Membership.mem ((convexHull R) s) (t.centerMass w z)
  -/
  rw [← centerMass_neg_left]
  /-
    R : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    s : Set E
    w : ι → R
    z : ι → E
    t : Finset ι
    hw₀ : ∀ (i : ι), Membership.mem t i → LE.le (w i) 0
    hws : LT.lt (t.sum fun i => w i) 0
    hz : ∀ (i : ι), Membership.mem t i → Membership.mem s (z i)
    ⊢ Membership.mem ((convexHull R) s) (t.centerMass (Neg.neg w) z)
  -/
  exact Finset.centerMass_mem_convexHull _ (fun _i hi ↦ neg_nonneg.2 <| hw₀ _ hi) (by simpa) hz
  /-
    🎉 no goals
  -/


/-- A refinement of `Finset.centerMass_mem_convexHull` when the indexed family is a `Finset` of
the space. -/
theorem Finset.centerMass_id_mem_convexHull (t : Finset E) {w : E → R} (hw₀ : ∀ i ∈ t, 0 ≤ w i)
    (hws : 0 < ∑ i ∈ t, w i) : t.centerMass w id ∈ convexHull R (t : Set E) :=
  t.centerMass_mem_convexHull hw₀ hws fun _ => mem_coe.2


/-- A version of `Finset.centerMass_mem_convexHull` for when the weights are nonpositive. -/
lemma Finset.centerMass_id_mem_convexHull_of_nonpos (t : Finset E) {w : E → R}
    (hw₀ : ∀ i ∈ t, w i ≤ 0) (hws : ∑ i ∈ t, w i < 0) :
    t.centerMass w id ∈ convexHull R (t : Set E) :=
  t.centerMass_mem_convexHull_of_nonpos hw₀ hws fun _ ↦ mem_coe.2


theorem affineCombination_eq_centerMass {ι : Type*} {t : Finset ι} {p : ι → E} {w : ι → R}
    (hw₂ : ∑ i ∈ t, w i = 1) : t.affineCombination R p w = centerMass t w p := by
  rw [affineCombination_eq_weightedVSubOfPoint_vadd_of_sum_eq_one _ w _ hw₂ (0 : E),
    Finset.weightedVSubOfPoint_apply, vadd_eq_add, add_zero, t.centerMass_eq_of_sum_1 _ hw₂]
  /-
    R : Type u_1
    E : Type u_3
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    ι : Type u_8
    t : Finset ι
    p : ι → E
    w : ι → R
    hw₂ : Eq (t.sum fun i => w i) 1
    ⊢ Eq (t.sum fun i => HSMul.hSMul (w i) (VSub.vsub (p i) 0)) (t.sum fun i => HS …
  -/
  simp_rw [vsub_eq_sub, sub_zero]
  /-
    🎉 no goals
  -/


theorem affineCombination_mem_convexHull {s : Finset ι} {v : ι → E} {w : ι → R}
    (hw₀ : ∀ i ∈ s, 0 ≤ w i) (hw₁ : s.sum w = 1) :
    s.affineCombination R v w ∈ convexHull R (range v) := by
  /-
    R : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    s : Finset ι
    v : ι → E
    w : ι → R
    hw₀ : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
    hw₁ : Eq (s.sum w) 1
    ⊢ Membership.mem ((convexHull R) (Set.range v)) ((Finset.affineCombination R s …
  -/
  rw [affineCombination_eq_centerMass hw₁]
  /-
    R : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    s : Finset ι
    v : ι → E
    w : ι → R
    hw₀ : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
    hw₁ : Eq (s.sum w) 1
    ⊢ Membership.mem ((convexHull R) (Set.range v)) (s.centerMass w v)
  -/
  apply s.centerMass_mem_convexHull hw₀
    /-
      case hws
      R : Type u_1
      E : Type u_3
      ι : Type u_5
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      s : Finset ι
      v : ι → E
      w : ι → R
      hw₀ : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
      hw₁ : Eq (s.sum w) 1
      ⊢ LT.lt 0 (s.sum fun i => w i)
    -/
  · simp [hw₁]
    /-
      🎉 no goals
    -/
    /-
      case hz
      R : Type u_1
      E : Type u_3
      ι : Type u_5
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      s : Finset ι
      v : ι → E
      w : ι → R
      hw₀ : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
      hw₁ : Eq (s.sum w) 1
      ⊢ ∀ (i : ι), Membership.mem s i → Membership.mem (Set.range v) (v i)
    -/
  · simp
    /-
      🎉 no goals
    -/


/-- The centroid can be regarded as a center of mass. -/
@[simp]
theorem Finset.centroid_eq_centerMass (s : Finset ι) (hs : s.Nonempty) (p : ι → E) :
    s.centroid R p = s.centerMass (s.centroidWeights R) p :=
  affineCombination_eq_centerMass (s.sum_centroidWeights_eq_one_of_nonempty R hs)


theorem Finset.centroid_mem_convexHull (s : Finset E) (hs : s.Nonempty) :
    s.centroid R id ∈ convexHull R (s : Set E) := by
  /-
    R : Type u_1
    E : Type u_3
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    s : Finset E
    hs : s.Nonempty
    ⊢ Membership.mem ((convexHull R) ↑s) (Finset.centroid R s id)
  -/
  rw [s.centroid_eq_centerMass hs]
  /-
    R : Type u_1
    E : Type u_3
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    s : Finset E
    hs : s.Nonempty
    ⊢ Membership.mem ((convexHull R) ↑s) (s.centerMass (Finset.centroidWeights R s …
  -/
  apply s.centerMass_id_mem_convexHull
    /-
      case hw₀
      R : Type u_1
      E : Type u_3
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      s : Finset E
      hs : s.Nonempty
      ⊢ ∀ (i : E), Membership.mem s i → LE.le 0 (Finset.centroidWeights R s i)
    -/
  · simp only [inv_nonneg, imp_true_iff, Nat.cast_nonneg, Finset.centroidWeights_apply]
    /-
      🎉 no goals
    -/
    /-
      case hws
      R : Type u_1
      E : Type u_3
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      s : Finset E
      hs : s.Nonempty
      ⊢ LT.lt 0 (s.sum fun i => Finset.centroidWeights R s i)
    -/
  · have hs_card : (#s : R) ≠ 0 := by simp [Finset.nonempty_iff_ne_empty.mp hs]
    simp only [hs_card, Finset.sum_const, nsmul_eq_mul, mul_inv_cancel₀, Ne, not_false_iff,
      Finset.centroidWeights_apply, zero_lt_one]


theorem convexHull_range_eq_exists_affineCombination (v : ι → E) : convexHull R (range v) =
    { x | ∃ (s : Finset ι) (w : ι → R), (∀ i ∈ s, 0 ≤ w i) ∧ s.sum w = 1 ∧
      s.affineCombination R v w = x } := by
  classical
  refine Subset.antisymm (convexHull_min ?_ ?_) ?_
  · intro x hx
    obtain ⟨i, hi⟩ := Set.mem_range.mp hx
    exact ⟨{i}, Function.const ι (1 : R), by simp, by simp, by simp [hi]⟩
  · rintro x ⟨s, w, hw₀, hw₁, rfl⟩ y ⟨s', w', hw₀', hw₁', rfl⟩ a b ha hb hab
    let W : ι → R := fun i => (if i ∈ s then a * w i else 0) + if i ∈ s' then b * w' i else 0
    have hW₁ : (s ∪ s').sum W = 1 := by
      rw [sum_add_distrib, ← sum_subset subset_union_left,
        ← sum_subset subset_union_right, sum_ite_of_true,
        sum_ite_of_true, ← mul_sum, ← mul_sum, hw₁, hw₁', ← add_mul, hab,
        mul_one] <;> intros <;> simp_all
    refine ⟨s ∪ s', W, ?_, hW₁, ?_⟩
    · rintro i -
      by_cases hi : i ∈ s <;> by_cases hi' : i ∈ s' <;>
        simp [W, hi, hi', add_nonneg, mul_nonneg ha (hw₀ i _), mul_nonneg hb (hw₀' i _)]
    · simp_rw [W, affineCombination_eq_linear_combination (s ∪ s') v _ hW₁,
        affineCombination_eq_linear_combination s v w hw₁,
        affineCombination_eq_linear_combination s' v w' hw₁', add_smul, sum_add_distrib]
      rw [← sum_subset subset_union_left, ← sum_subset subset_union_right]
      · simp only [ite_smul, sum_ite_of_true fun _ hi => hi, mul_smul, ← smul_sum]
      · intro i _ hi'
        simp [hi']
      · intro i _ hi'
        simp [hi']
  · rintro x ⟨s, w, hw₀, hw₁, rfl⟩
    exact affineCombination_mem_convexHull hw₀ hw₁


/--
Convex hull of `s` is equal to the set of all centers of masses of `Finset`s `t`, `z '' t ⊆ s`.
For universe reasons, you shouldn't use this lemma to prove that a given center of mass belongs
to the convex hull. Use convexity of the convex hull instead.
-/
theorem convexHull_eq (s : Set E) : convexHull R s =
    { x : E | ∃ (ι : Type) (t : Finset ι) (w : ι → R) (z : ι → E), (∀ i ∈ t, 0 ≤ w i) ∧
      ∑ i ∈ t, w i = 1 ∧ (∀ i ∈ t, z i ∈ s) ∧ t.centerMass w z = x } := by
  /-
    R : Type u_1
    E : Type u_3
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    s : Set E
    ⊢ Eq ((convexHull R) s) (setOf fun x => Exists fun ι => Exists fun t => Exists …
  -/
  refine Subset.antisymm (convexHull_min ?_ ?_) ?_
    /-
      case refine_1
      R : Type u_1
      E : Type u_3
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      s : Set E
      ⊢ HasSubset.Subset s (setOf fun x => Exists fun ι => Exists fun t => Exists fu …
    -/
  · intro x hx
    use PUnit, {PUnit.unit}, fun _ => 1, fun _ => x, fun _ _ => zero_le_one, sum_singleton _ _,
      fun _ _ => hx
    /-
      case right
      R : Type u_1
      E : Type u_3
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      s : Set E
      x : E
      hx : Membership.mem s x
      ⊢ Eq ((Singleton.singleton PUnit.unit).centerMass (fun x => 1) fun x_1 => x) x
    -/
    simp only [Finset.centerMass, Finset.sum_singleton, inv_one, one_smul]
    /-
      🎉 no goals
    -/
  · rintro x ⟨ι, sx, wx, zx, hwx₀, hwx₁, hzx, rfl⟩ y ⟨ι', sy, wy, zy, hwy₀, hwy₁, hzy, rfl⟩ a b ha
      hb hab
    /-
      case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
      R : Type u_1
      E : Type u_3
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      s : Set E
      ι : Type
      sx : Finset ι
      wx : ι → R
      zx : ι → E
      hwx₀ : ∀ (i : ι), Membership.mem sx i → LE.le 0 (wx i)
      hwx₁ : Eq (sx.sum fun i => wx i) 1
      hzx : ∀ (i : ι), Membership.mem sx i → Membership.mem s (zx i)
      ι' : Type
      sy : Finset ι'
      wy : ι' → R
      zy : ι' → E
      hwy₀ : ∀ (i : ι'), Membership.mem sy i → LE.le 0 (wy i)
      hwy₁ : Eq (sy.sum fun i => wy i) 1
      hzy : ∀ (i : ι'), Membership.mem sy i → Membership.mem s (zy i)
      a b : R
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ Membership.mem (setOf fun x => Exists fun ι => Exists fun t => Exists fun w  …
    -/
    rw [Finset.centerMass_segment' _ _ _ _ _ _ hwx₁ hwy₁ _ _ hab]
    /-
      case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
      R : Type u_1
      E : Type u_3
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      s : Set E
      ι : Type
      sx : Finset ι
      wx : ι → R
      zx : ι → E
      hwx₀ : ∀ (i : ι), Membership.mem sx i → LE.le 0 (wx i)
      hwx₁ : Eq (sx.sum fun i => wx i) 1
      hzx : ∀ (i : ι), Membership.mem sx i → Membership.mem s (zx i)
      ι' : Type
      sy : Finset ι'
      wy : ι' → R
      zy : ι' → E
      hwy₀ : ∀ (i : ι'), Membership.mem sy i → LE.le 0 (wy i)
      hwy₁ : Eq (sy.sum fun i => wy i) 1
      hzy : ∀ (i : ι'), Membership.mem sy i → Membership.mem s (zy i)
      a b : R
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ Membership.mem (setOf fun x => Exists fun ι => Exists fun t => Exists fun w  …
    -/
    refine ⟨_, _, _, _, ?_, ?_, ?_, rfl⟩
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
        R : Type u_1
        E : Type u_3
        inst✝² : LinearOrderedField R
        inst✝¹ : AddCommGroup E
        inst✝ : Module R E
        s : Set E
        ι : Type
        sx : Finset ι
        wx : ι → R
        zx : ι → E
        hwx₀ : ∀ (i : ι), Membership.mem sx i → LE.le 0 (wx i)
        hwx₁ : Eq (sx.sum fun i => wx i) 1
        hzx : ∀ (i : ι), Membership.mem sx i → Membership.mem s (zx i)
        ι' : Type
        sy : Finset ι'
        wy : ι' → R
        zy : ι' → E
        hwy₀ : ∀ (i : ι'), Membership.mem sy i → LE.le 0 (wy i)
        hwy₁ : Eq (sy.sum fun i => wy i) 1
        hzy : ∀ (i : ι'), Membership.mem sy i → Membership.mem s (zy i)
        a b : R
        ha : LE.le 0 a
        hb : LE.le 0 b
        hab : Eq (HAdd.hAdd a b) 1
        ⊢ ∀ (i : Sum ι ι'), Membership.mem (sx.disjSum sy) i → LE.le 0 (Sum.elim (fun  …
      -/
    · rintro i hi
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
        R : Type u_1
        E : Type u_3
        inst✝² : LinearOrderedField R
        inst✝¹ : AddCommGroup E
        inst✝ : Module R E
        s : Set E
        ι : Type
        sx : Finset ι
        wx : ι → R
        zx : ι → E
        hwx₀ : ∀ (i : ι), Membership.mem sx i → LE.le 0 (wx i)
        hwx₁ : Eq (sx.sum fun i => wx i) 1
        hzx : ∀ (i : ι), Membership.mem sx i → Membership.mem s (zx i)
        ι' : Type
        sy : Finset ι'
        wy : ι' → R
        zy : ι' → E
        hwy₀ : ∀ (i : ι'), Membership.mem sy i → LE.le 0 (wy i)
        hwy₁ : Eq (sy.sum fun i => wy i) 1
        hzy : ∀ (i : ι'), Membership.mem sy i → Membership.mem s (zy i)
        a b : R
        ha : LE.le 0 a
        hb : LE.le 0 b
        hab : Eq (HAdd.hAdd a b) 1
        i : Sum ι ι'
        hi : Membership.mem (sx.disjSum sy) i
        ⊢ LE.le 0 (Sum.elim (fun i => HMul.hMul a (wx i)) (fun j => HMul.hMul b (wy j) …
      -/
      rw [Finset.mem_disjSum] at hi
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
        R : Type u_1
        E : Type u_3
        inst✝² : LinearOrderedField R
        inst✝¹ : AddCommGroup E
        inst✝ : Module R E
        s : Set E
        ι : Type
        sx : Finset ι
        wx : ι → R
        zx : ι → E
        hwx₀ : ∀ (i : ι), Membership.mem sx i → LE.le 0 (wx i)
        hwx₁ : Eq (sx.sum fun i => wx i) 1
        hzx : ∀ (i : ι), Membership.mem sx i → Membership.mem s (zx i)
        ι' : Type
        sy : Finset ι'
        wy : ι' → R
        zy : ι' → E
        hwy₀ : ∀ (i : ι'), Membership.mem sy i → LE.le 0 (wy i)
        hwy₁ : Eq (sy.sum fun i => wy i) 1
        hzy : ∀ (i : ι'), Membership.mem sy i → Membership.mem s (zy i)
        a b : R
        ha : LE.le 0 a
        hb : LE.le 0 b
        hab : Eq (HAdd.hAdd a b) 1
        i : Sum ι ι'
        hi : Or (Exists fun a => And (Membership.mem sx a) (Eq (Sum.inl a) i)) (Exists …
        ⊢ LE.le 0 (Sum.elim (fun i => HMul.hMul a (wx i)) (fun j => HMul.hMul b (wy j) …
      -/
      rcases hi with (⟨j, hj, rfl⟩ | ⟨j, hj, rfl⟩) <;> simp only [Sum.elim_inl, Sum.elim_inr] <;>
        /-
          case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
          R : Type u_1
          E : Type u_3
          inst✝² : LinearOrderedField R
          inst✝¹ : AddCommGroup E
          inst✝ : Module R E
          s : Set E
          ι : Type
          sx : Finset ι
          wx : ι → R
          zx : ι → E
          hwx₀ : ∀ (i : ι), Membership.mem sx i → LE.le 0 (wx i)
          hwx₁ : Eq (sx.sum fun i => wx i) 1
          hzx : ∀ (i : ι), Membership.mem sx i → Membership.mem s (zx i)
          ι' : Type
          sy : Finset ι'
          wy : ι' → R
          zy : ι' → E
          hwy₀ : ∀ (i : ι'), Membership.mem sy i → LE.le 0 (wy i)
          hwy₁ : Eq (sy.sum fun i => wy i) 1
          hzy : ∀ (i : ι'), Membership.mem sy i → Membership.mem s (zy i)
          a b : R
          ha : LE.le 0 a
          hb : LE.le 0 b
          hab : Eq (HAdd.hAdd a b) 1
          j : ι
          hj : Membership.mem sx j
          ⊢ LE.le 0 (HMul.hMul a (wx j))
        -/
        /-
          🎉 no goals
        -/
        apply_rules [mul_nonneg, hwx₀, hwy₀]
        /-
          🎉 no goals
        -/
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
        R : Type u_1
        E : Type u_3
        inst✝² : LinearOrderedField R
        inst✝¹ : AddCommGroup E
        inst✝ : Module R E
        s : Set E
        ι : Type
        sx : Finset ι
        wx : ι → R
        zx : ι → E
        hwx₀ : ∀ (i : ι), Membership.mem sx i → LE.le 0 (wx i)
        hwx₁ : Eq (sx.sum fun i => wx i) 1
        hzx : ∀ (i : ι), Membership.mem sx i → Membership.mem s (zx i)
        ι' : Type
        sy : Finset ι'
        wy : ι' → R
        zy : ι' → E
        hwy₀ : ∀ (i : ι'), Membership.mem sy i → LE.le 0 (wy i)
        hwy₁ : Eq (sy.sum fun i => wy i) 1
        hzy : ∀ (i : ι'), Membership.mem sy i → Membership.mem s (zy i)
        a b : R
        ha : LE.le 0 a
        hb : LE.le 0 b
        hab : Eq (HAdd.hAdd a b) 1
        ⊢ Eq ((sx.disjSum sy).sum fun i => Sum.elim (fun i => HMul.hMul a (wx i)) (fun …
      -/
    · simp [Finset.sum_sum_elim, ← mul_sum, *]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
        R : Type u_1
        E : Type u_3
        inst✝² : LinearOrderedField R
        inst✝¹ : AddCommGroup E
        inst✝ : Module R E
        s : Set E
        ι : Type
        sx : Finset ι
        wx : ι → R
        zx : ι → E
        hwx₀ : ∀ (i : ι), Membership.mem sx i → LE.le 0 (wx i)
        hwx₁ : Eq (sx.sum fun i => wx i) 1
        hzx : ∀ (i : ι), Membership.mem sx i → Membership.mem s (zx i)
        ι' : Type
        sy : Finset ι'
        wy : ι' → R
        zy : ι' → E
        hwy₀ : ∀ (i : ι'), Membership.mem sy i → LE.le 0 (wy i)
        hwy₁ : Eq (sy.sum fun i => wy i) 1
        hzy : ∀ (i : ι'), Membership.mem sy i → Membership.mem s (zy i)
        a b : R
        ha : LE.le 0 a
        hb : LE.le 0 b
        hab : Eq (HAdd.hAdd a b) 1
        ⊢ ∀ (i : Sum ι ι'), Membership.mem (sx.disjSum sy) i → Membership.mem s (Sum.e …
      -/
    · intro i hi
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
        R : Type u_1
        E : Type u_3
        inst✝² : LinearOrderedField R
        inst✝¹ : AddCommGroup E
        inst✝ : Module R E
        s : Set E
        ι : Type
        sx : Finset ι
        wx : ι → R
        zx : ι → E
        hwx₀ : ∀ (i : ι), Membership.mem sx i → LE.le 0 (wx i)
        hwx₁ : Eq (sx.sum fun i => wx i) 1
        hzx : ∀ (i : ι), Membership.mem sx i → Membership.mem s (zx i)
        ι' : Type
        sy : Finset ι'
        wy : ι' → R
        zy : ι' → E
        hwy₀ : ∀ (i : ι'), Membership.mem sy i → LE.le 0 (wy i)
        hwy₁ : Eq (sy.sum fun i => wy i) 1
        hzy : ∀ (i : ι'), Membership.mem sy i → Membership.mem s (zy i)
        a b : R
        ha : LE.le 0 a
        hb : LE.le 0 b
        hab : Eq (HAdd.hAdd a b) 1
        i : Sum ι ι'
        hi : Membership.mem (sx.disjSum sy) i
        ⊢ Membership.mem s (Sum.elim zx zy i)
      -/
      rw [Finset.mem_disjSum] at hi
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
        R : Type u_1
        E : Type u_3
        inst✝² : LinearOrderedField R
        inst✝¹ : AddCommGroup E
        inst✝ : Module R E
        s : Set E
        ι : Type
        sx : Finset ι
        wx : ι → R
        zx : ι → E
        hwx₀ : ∀ (i : ι), Membership.mem sx i → LE.le 0 (wx i)
        hwx₁ : Eq (sx.sum fun i => wx i) 1
        hzx : ∀ (i : ι), Membership.mem sx i → Membership.mem s (zx i)
        ι' : Type
        sy : Finset ι'
        wy : ι' → R
        zy : ι' → E
        hwy₀ : ∀ (i : ι'), Membership.mem sy i → LE.le 0 (wy i)
        hwy₁ : Eq (sy.sum fun i => wy i) 1
        hzy : ∀ (i : ι'), Membership.mem sy i → Membership.mem s (zy i)
        a b : R
        ha : LE.le 0 a
        hb : LE.le 0 b
        hab : Eq (HAdd.hAdd a b) 1
        i : Sum ι ι'
        hi : Or (Exists fun a => And (Membership.mem sx a) (Eq (Sum.inl a) i)) (Exists …
        ⊢ Membership.mem s (Sum.elim zx zy i)
      -/
                                                       /-
                                                         🎉 no goals
                                                       -/
      rcases hi with (⟨j, hj, rfl⟩ | ⟨j, hj, rfl⟩) <;> apply_rules [hzx, hzy]
                                                       /-
                                                         🎉 no goals
                                                       -/
    /-
      case refine_3
      R : Type u_1
      E : Type u_3
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      s : Set E
      ⊢ HasSubset.Subset (setOf fun x => Exists fun ι => Exists fun t => Exists fun  …
    -/
  · rintro _ ⟨ι, t, w, z, hw₀, hw₁, hz, rfl⟩
    /-
      case refine_3.intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      E : Type u_3
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      s : Set E
      ι : Type
      t : Finset ι
      w : ι → R
      z : ι → E
      hw₀ : ∀ (i : ι), Membership.mem t i → LE.le 0 (w i)
      hw₁ : Eq (t.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem t i → Membership.mem s (z i)
      ⊢ Membership.mem ((convexHull R) s) (t.centerMass w z)
    -/
    exact t.centerMass_mem_convexHull hw₀ (hw₁.symm ▸ zero_lt_one) hz
    /-
      🎉 no goals
    -/


/-- Universe polymorphic version of the reverse implication of `mem_convexHull_iff_exists_fintype`.
-/
lemma mem_convexHull_of_exists_fintype {s : Set E} {x : E} [Fintype ι] (w : ι → R) (z : ι → E)
    (hw₀ : ∀ i, 0 ≤ w i) (hw₁ : ∑ i, w i = 1) (hz : ∀ i, z i ∈ s) (hx : ∑ i, w i • z i = x) :
    x ∈ convexHull R s := by
  /-
    R : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup E
    inst✝¹ : Module R E
    s : Set E
    x : E
    inst✝ : Fintype ι
    w : ι → R
    z : ι → E
    hw₀ : ∀ (i : ι), LE.le 0 (w i)
    hw₁ : Eq (Finset.univ.sum fun i => w i) 1
    hz : ∀ (i : ι), Membership.mem s (z i)
    hx : Eq (Finset.univ.sum fun i => HSMul.hSMul (w i) (z i)) x
    ⊢ Membership.mem ((convexHull R) s) x
  -/
  rw [← hx, ← centerMass_eq_of_sum_1 _ _ hw₁]
  /-
    R : Type u_1
    E : Type u_3
    ι : Type u_5
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup E
    inst✝¹ : Module R E
    s : Set E
    x : E
    inst✝ : Fintype ι
    w : ι → R
    z : ι → E
    hw₀ : ∀ (i : ι), LE.le 0 (w i)
    hw₁ : Eq (Finset.univ.sum fun i => w i) 1
    hz : ∀ (i : ι), Membership.mem s (z i)
    hx : Eq (Finset.univ.sum fun i => HSMul.hSMul (w i) (z i)) x
    ⊢ Membership.mem ((convexHull R) s) (Finset.univ.centerMass w z)
  -/
  exact centerMass_mem_convexHull _ (by simpa using hw₀) (by simp [hw₁]) (by simpa using hz)
  /-
    🎉 no goals
  -/


/-- The convex hull of `s` is equal to the set of centers of masses of finite families of points in
`s`.

For universe reasons, you shouldn't use this lemma to prove that a given center of mass belongs
to the convex hull. Use `mem_convexHull_of_exists_fintype` of the convex hull instead. -/
lemma mem_convexHull_iff_exists_fintype {s : Set E} {x : E} :
    x ∈ convexHull R s ↔ ∃ (ι : Type) (_ : Fintype ι) (w : ι → R) (z : ι → E), (∀ i, 0 ≤ w i) ∧
      ∑ i, w i = 1 ∧ (∀ i, z i ∈ s) ∧ ∑ i, w i • z i = x := by
  /-
    R : Type u_1
    E : Type u_3
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    s : Set E
    x : E
    ⊢ Iff (Membership.mem ((convexHull R) s) x) (Exists fun ι => Exists fun x_1 => …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      E : Type u_3
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      s : Set E
      x : E
      ⊢ Membership.mem ((convexHull R) s) x → Exists fun ι => Exists fun x_1 => Exis …
    -/
  · simp only [convexHull_eq, mem_setOf_eq]
    /-
      case mp
      R : Type u_1
      E : Type u_3
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      s : Set E
      x : E
      ⊢ (Exists fun ι => Exists fun t => Exists fun w => Exists fun z => And (∀ (i : …
    -/
    rintro ⟨ι, t, w, z, h⟩
    /-
      case mp.intro.intro.intro.intro
      R : Type u_1
      E : Type u_3
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      s : Set E
      x : E
      ι : Type
      t : Finset ι
      w : ι → R
      z : ι → E
      h : And (∀ (i : ι), Membership.mem t i → LE.le 0 (w i)) (And (Eq (t.sum fun i  …
      ⊢ Exists fun ι => Exists fun x_1 => Exists fun w => Exists fun z => And (∀ (i  …
    -/
    refine ⟨t, inferInstance, w ∘ (↑), z ∘ (↑), ?_⟩
    /-
      case mp.intro.intro.intro.intro
      R : Type u_1
      E : Type u_3
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      s : Set E
      x : E
      ι : Type
      t : Finset ι
      w : ι → R
      z : ι → E
      h : And (∀ (i : ι), Membership.mem t i → LE.le 0 (w i)) (And (Eq (t.sum fun i  …
      ⊢ And (∀ (i : Subtype fun x => Membership.mem t x), LE.le 0 (Function.comp w S …
    -/
    simpa [← sum_attach t, centerMass_eq_of_sum_1 _ _ h.2.1] using h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      E : Type u_3
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      s : Set E
      x : E
      ⊢ (Exists fun ι => Exists fun x_1 => Exists fun w => Exists fun z => And (∀ (i …
    -/
  · rintro ⟨ι, _, w, z, hw₀, hw₁, hz, hx⟩
    /-
      case mpr.intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      E : Type u_3
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      s : Set E
      x : E
      ι : Type
      w✝ : Fintype ι
      w : ι → R
      z : ι → E
      hw₀ : ∀ (i : ι), LE.le 0 (w i)
      hw₁ : Eq (Finset.univ.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s (z i)
      hx : Eq (Finset.univ.sum fun i => HSMul.hSMul (w i) (z i)) x
      ⊢ Membership.mem ((convexHull R) s) x
    -/
    exact mem_convexHull_of_exists_fintype w z hw₀ hw₁ hz hx
    /-
      🎉 no goals
    -/


theorem Finset.convexHull_eq (s : Finset E) : convexHull R ↑s =
    { x : E | ∃ w : E → R, (∀ y ∈ s, 0 ≤ w y) ∧ ∑ y ∈ s, w y = 1 ∧ s.centerMass w id = x } := by
  classical
  refine Set.Subset.antisymm (convexHull_min ?_ ?_) ?_
  · intro x hx
    rw [Finset.mem_coe] at hx
    refine ⟨_, ?_, ?_, Finset.centerMass_ite_eq _ _ _ hx⟩
    · intros
      split_ifs
      exacts [zero_le_one, le_refl 0]
    · rw [Finset.sum_ite_eq, if_pos hx]
  · rintro x ⟨wx, hwx₀, hwx₁, rfl⟩ y ⟨wy, hwy₀, hwy₁, rfl⟩ a b ha hb hab
    rw [Finset.centerMass_segment _ _ _ _ hwx₁ hwy₁ _ _ hab]
    refine ⟨_, ?_, ?_, rfl⟩
    · rintro i hi
      apply_rules [add_nonneg, mul_nonneg, hwx₀, hwy₀]
    · simp only [Finset.sum_add_distrib, ← mul_sum, mul_one, *]
  · rintro _ ⟨w, hw₀, hw₁, rfl⟩
    exact
      s.centerMass_mem_convexHull (fun x hx => hw₀ _ hx) (hw₁.symm ▸ zero_lt_one) fun x hx => hx


theorem Finset.mem_convexHull {s : Finset E} {x : E} : x ∈ convexHull R (s : Set E) ↔
    ∃ w : E → R, (∀ y ∈ s, 0 ≤ w y) ∧ ∑ y ∈ s, w y = 1 ∧ s.centerMass w id = x := by
  /-
    R : Type u_1
    E : Type u_3
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    s : Finset E
    x : E
    ⊢ Iff (Membership.mem ((convexHull R) ↑s) x) (Exists fun w => And (∀ (y : E),  …
  -/
  rw [Finset.convexHull_eq, Set.mem_setOf_eq]
  /-
    🎉 no goals
  -/


/-- This is a version of `Finset.mem_convexHull` stated without `Finset.centerMass`. -/
lemma Finset.mem_convexHull' {s : Finset E} {x : E} :
    x ∈ convexHull R (s : Set E) ↔
      ∃ w : E → R, (∀ y ∈ s, 0 ≤ w y) ∧ ∑ y ∈ s, w y = 1 ∧ ∑ y ∈ s, w y • y = x := by
  /-
    R : Type u_1
    E : Type u_3
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    s : Finset E
    x : E
    ⊢ Iff (Membership.mem ((convexHull R) ↑s) x) (Exists fun w => And (∀ (y : E),  …
  -/
  rw [mem_convexHull]
  /-
    R : Type u_1
    E : Type u_3
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    s : Finset E
    x : E
    ⊢ Iff (Exists fun w => And (∀ (y : E), Membership.mem s y → LE.le 0 (w y)) (An …
  -/
  refine exists_congr fun w ↦ and_congr_right' <| and_congr_right fun hw ↦ ?_
  /-
    R : Type u_1
    E : Type u_3
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    s : Finset E
    x : E
    w : E → R
    hw : Eq (s.sum fun y => w y) 1
    ⊢ Iff (Eq (s.centerMass w id) x) (Eq (s.sum fun y => HSMul.hSMul (w y) y) x)
  -/
  simp_rw [centerMass_eq_of_sum_1 _ _ hw, id_eq]
  /-
    🎉 no goals
  -/


theorem Set.Finite.convexHull_eq {s : Set E} (hs : s.Finite) : convexHull R s =
    { x : E | ∃ w : E → R, (∀ y ∈ s, 0 ≤ w y) ∧ ∑ y ∈ hs.toFinset, w y = 1 ∧
      hs.toFinset.centerMass w id = x } := by
  simpa only [Set.Finite.coe_toFinset, Set.Finite.mem_toFinset, exists_prop] using
    hs.toFinset.convexHull_eq


/-- A weak version of Carathéodory's theorem. -/
theorem convexHull_eq_union_convexHull_finite_subsets (s : Set E) :
    convexHull R s = ⋃ (t : Finset E) (_ : ↑t ⊆ s), convexHull R ↑t := by
  classical
  refine Subset.antisymm ?_ ?_
  · rw [_root_.convexHull_eq]
    rintro x ⟨ι, t, w, z, hw₀, hw₁, hz, rfl⟩
    simp only [mem_iUnion]
    refine ⟨t.image z, ?_, ?_⟩
    · rw [coe_image, Set.image_subset_iff]
      exact hz
    · apply t.centerMass_mem_convexHull hw₀
      · simp only [hw₁, zero_lt_one]
      · exact fun i hi => Finset.mem_coe.2 (Finset.mem_image_of_mem _ hi)
  · exact iUnion_subset fun i => iUnion_subset convexHull_mono


theorem mk_mem_convexHull_prod {t : Set F} {x : E} {y : F} (hx : x ∈ convexHull R s)
    (hy : y ∈ convexHull R t) : (x, y) ∈ convexHull R (s ×ˢ t) := by
  /-
    R : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁴ : LinearOrderedField R
    inst✝³ : AddCommGroup E
    inst✝² : AddCommGroup F
    inst✝¹ : Module R E
    inst✝ : Module R F
    s : Set E
    t : Set F
    x : E
    y : F
    hx : Membership.mem ((convexHull R) s) x
    hy : Membership.mem ((convexHull R) t) y
    ⊢ Membership.mem ((convexHull R) (SProd.sprod s t)) { fst := x, snd := y }
  -/
  rw [mem_convexHull_iff_exists_fintype] at hx hy ⊢
  /-
    R : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁴ : LinearOrderedField R
    inst✝³ : AddCommGroup E
    inst✝² : AddCommGroup F
    inst✝¹ : Module R E
    inst✝ : Module R F
    s : Set E
    t : Set F
    x : E
    y : F
    hx : Exists fun ι => Exists fun x_1 => Exists fun w => Exists fun z => And (∀  …
    hy : Exists fun ι => Exists fun x => Exists fun w => Exists fun z => And (∀ (i …
    ⊢ Exists fun ι => Exists fun x_1 => Exists fun w => Exists fun z => And (∀ (i  …
  -/
  obtain ⟨ι, _, w, f, hw₀, hw₁, hfs, hf⟩ := hx
  /-
    case intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁴ : LinearOrderedField R
    inst✝³ : AddCommGroup E
    inst✝² : AddCommGroup F
    inst✝¹ : Module R E
    inst✝ : Module R F
    s : Set E
    t : Set F
    x : E
    y : F
    hy : Exists fun ι => Exists fun x => Exists fun w => Exists fun z => And (∀ (i …
    ι : Type
    w✝ : Fintype ι
    w : ι → R
    f : ι → E
    hw₀ : ∀ (i : ι), LE.le 0 (w i)
    hw₁ : Eq (Finset.univ.sum fun i => w i) 1
    hfs : ∀ (i : ι), Membership.mem s (f i)
    hf : Eq (Finset.univ.sum fun i => HSMul.hSMul (w i) (f i)) x
    ⊢ Exists fun ι => Exists fun x_1 => Exists fun w => Exists fun z => And (∀ (i  …
  -/
  obtain ⟨κ, _, v, g, hv₀, hv₁, hgt, hg⟩ := hy
  have h_sum : ∑ i : ι × κ, w i.1 * v i.2 = 1 := by
    rw [Fintype.sum_prod_type, ← sum_mul_sum, hw₁, hv₁, mul_one]
  refine ⟨ι × κ, inferInstance, fun p => w p.1 * v p.2, fun p ↦ (f p.1, g p.2),
    fun p ↦ mul_nonneg (hw₀ _) (hv₀ _), h_sum, fun p ↦ ⟨hfs _, hgt _⟩, ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    R : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁴ : LinearOrderedField R
    inst✝³ : AddCommGroup E
    inst✝² : AddCommGroup F
    inst✝¹ : Module R E
    inst✝ : Module R F
    s : Set E
    t : Set F
    x : E
    y : F
    ι : Type
    w✝¹ : Fintype ι
    w : ι → R
    f : ι → E
    hw₀ : ∀ (i : ι), LE.le 0 (w i)
    hw₁ : Eq (Finset.univ.sum fun i => w i) 1
    hfs : ∀ (i : ι), Membership.mem s (f i)
    hf : Eq (Finset.univ.sum fun i => HSMul.hSMul (w i) (f i)) x
    κ : Type
    w✝ : Fintype κ
    v : κ → R
    g : κ → F
    hv₀ : ∀ (i : κ), LE.le 0 (v i)
    hv₁ : Eq (Finset.univ.sum fun i => v i) 1
    hgt : ∀ (i : κ), Membership.mem t (g i)
    hg : Eq (Finset.univ.sum fun i => HSMul.hSMul (v i) (g i)) y
    h_sum : Eq (Finset.univ.sum fun i => HMul.hMul (w i.1) (v i.2)) 1
    ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul ((fun p => HMul.hMul (w p.1) (v p.2 …
  -/
  ext
  · simp_rw [Prod.fst_sum, Prod.smul_mk, Fintype.sum_prod_type, mul_comm (w _), mul_smul,
      sum_comm (γ := ι), ← Fintype.sum_smul_sum, hv₁, one_smul, hf]
  · simp_rw [Prod.snd_sum, Prod.smul_mk, Fintype.sum_prod_type, mul_smul, ← Fintype.sum_smul_sum,
      hw₁, one_smul, hg]


@[simp]
theorem convexHull_prod (s : Set E) (t : Set F) :
    convexHull R (s ×ˢ t) = convexHull R s ×ˢ convexHull R t :=
  Subset.antisymm
      (convexHull_min (prod_mono (subset_convexHull _ _) <| subset_convexHull _ _) <|
        (convex_convexHull _ _).prod <| convex_convexHull _ _) <|
    prod_subset_iff.2 fun _ hx _ => mk_mem_convexHull_prod hx


theorem convexHull_add (s t : Set E) : convexHull R (s + t) = convexHull R s + convexHull R t := by
  /-
    R : Type u_1
    E : Type u_3
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    s t : Set E
    ⊢ Eq ((convexHull R) (HAdd.hAdd s t)) (HAdd.hAdd ((convexHull R) s) ((convexHu …
  -/
  simp_rw [← add_image_prod, ← IsLinearMap.isLinearMap_add.image_convexHull, convexHull_prod]
  /-
    🎉 no goals
  -/


/-- `convexHull` is an additive monoid morphism under pointwise addition. -/
@[simps]
def convexHullAddMonoidHom : Set E →+ Set E where
  toFun := convexHull R
  map_add' := convexHull_add
  map_zero' := convexHull_zero


theorem convexHull_sub (s t : Set E) : convexHull R (s - t) = convexHull R s - convexHull R t := by
  /-
    R : Type u_1
    E : Type u_3
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    s t : Set E
    ⊢ Eq ((convexHull R) (HSub.hSub s t)) (HSub.hSub ((convexHull R) s) ((convexHu …
  -/
  simp_rw [sub_eq_add_neg, convexHull_add, ← convexHull_neg]
  /-
    🎉 no goals
  -/


theorem convexHull_list_sum (l : List (Set E)) : convexHull R l.sum = (l.map <| convexHull R).sum :=
  map_list_sum (convexHullAddMonoidHom R E) l


theorem convexHull_multiset_sum (s : Multiset (Set E)) :
    convexHull R s.sum = (s.map <| convexHull R).sum :=
  map_multiset_sum (convexHullAddMonoidHom R E) s


theorem convexHull_sum {ι} (s : Finset ι) (t : ι → Set E) :
    convexHull R (∑ i ∈ s, t i) = ∑ i ∈ s, convexHull R (t i) :=
  map_sum (convexHullAddMonoidHom R E) _ _


open scoped Classical in
/-- `stdSimplex 𝕜 ι` is the convex hull of the canonical basis in `ι → 𝕜`. -/
theorem convexHull_basis_eq_stdSimplex :
    convexHull R (range fun i j : ι => if i = j then (1 : R) else 0) = stdSimplex R ι := by
  /-
    R : Type u_1
    ι : Type u_5
    inst✝¹ : LinearOrderedField R
    inst✝ : Fintype ι
    ⊢ Eq ((convexHull R) (Set.range fun i j => ite (Eq i j) 1 0)) (stdSimplex R ι)
  -/
  refine Subset.antisymm (convexHull_min ?_ (convex_stdSimplex R ι)) ?_
    /-
      case refine_1
      R : Type u_1
      ι : Type u_5
      inst✝¹ : LinearOrderedField R
      inst✝ : Fintype ι
      ⊢ HasSubset.Subset (Set.range fun i j => ite (Eq i j) 1 0) (stdSimplex R ι)
    -/
  · rintro _ ⟨i, rfl⟩
    /-
      case refine_1.intro
      R : Type u_1
      ι : Type u_5
      inst✝¹ : LinearOrderedField R
      inst✝ : Fintype ι
      i : ι
      ⊢ Membership.mem (stdSimplex R ι) ((fun i j => ite (Eq i j) 1 0) i)
    -/
    exact ite_eq_mem_stdSimplex R i
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      ι : Type u_5
      inst✝¹ : LinearOrderedField R
      inst✝ : Fintype ι
      ⊢ HasSubset.Subset (stdSimplex R ι) ((convexHull R) (Set.range fun i j => ite  …
    -/
  · rintro w ⟨hw₀, hw₁⟩
    /-
      case refine_2.intro
      R : Type u_1
      ι : Type u_5
      inst✝¹ : LinearOrderedField R
      inst✝ : Fintype ι
      w : ι → R
      hw₀ : ∀ (x : ι), LE.le 0 (w x)
      hw₁ : Eq (Finset.univ.sum fun x => w x) 1
      ⊢ Membership.mem ((convexHull R) (Set.range fun i j => ite (Eq i j) 1 0)) w
    -/
    rw [pi_eq_sum_univ w, ← Finset.univ.centerMass_eq_of_sum_1 _ hw₁]
    exact Finset.univ.centerMass_mem_convexHull (fun i _ => hw₀ i) (hw₁.symm ▸ zero_lt_one)
      fun i _ => mem_range_self i


/-- The convex hull of a finite set is the image of the standard simplex in `s → ℝ`
under the linear map sending each function `w` to `∑ x ∈ s, w x • x`.

Since we have no sums over finite sets, we use sum over `@Finset.univ _ hs.fintype`.
The map is defined in terms of operations on `(s → ℝ) →ₗ[ℝ] ℝ` so that later we will not need
to prove that this map is linear. -/
theorem Set.Finite.convexHull_eq_image {s : Set E} (hs : s.Finite) : convexHull R s =
    haveI := hs.fintype
    (⇑(∑ x : s, (@LinearMap.proj R s _ (fun _ => R) _ _ x).smulRight x.1)) '' stdSimplex R s := by
  classical
  letI := hs.fintype
  rw [← convexHull_basis_eq_stdSimplex, LinearMap.image_convexHull, ← Set.range_comp]
  apply congr_arg
  simp_rw [Function.comp_def]
  convert Subtype.range_coe.symm
  simp [LinearMap.sum_apply, ite_smul, Finset.filter_eq, Finset.mem_univ]


/-- All values of a function `f ∈ stdSimplex 𝕜 ι` belong to `[0, 1]`. -/
theorem mem_Icc_of_mem_stdSimplex (hf : f ∈ stdSimplex R ι) (x) : f x ∈ Icc (0 : R) 1 :=
  ⟨hf.1 x, hf.2 ▸ Finset.single_le_sum (fun y _ => hf.1 y) (Finset.mem_univ x)⟩


/-- The convex hull of an affine basis is the intersection of the half-spaces defined by the
corresponding barycentric coordinates. -/
theorem AffineBasis.convexHull_eq_nonneg_coord {ι : Type*} (b : AffineBasis ι R E) :
    convexHull R (range b) = { x | ∀ i, 0 ≤ b.coord i x } := by
  /-
    R : Type u_1
    E : Type u_3
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    ι : Type u_8
    b : AffineBasis ι R E
    ⊢ Eq ((convexHull R) (Set.range ⇑b)) (setOf fun x => ∀ (i : ι), LE.le 0 ((b.co …
  -/
  rw [convexHull_range_eq_exists_affineCombination]
  /-
    R : Type u_1
    E : Type u_3
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    ι : Type u_8
    b : AffineBasis ι R E
    ⊢ Eq (setOf fun x => Exists fun s => Exists fun w => And (∀ (i : ι), Membershi …
  -/
  ext x
  /-
    case h
    R : Type u_1
    E : Type u_3
    inst✝² : LinearOrderedField R
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    ι : Type u_8
    b : AffineBasis ι R E
    x : E
    ⊢ Iff (Membership.mem (setOf fun x => Exists fun s => Exists fun w => And (∀ ( …
  -/
  refine ⟨?_, fun hx => ?_⟩
    /-
      case h.refine_1
      R : Type u_1
      E : Type u_3
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      ι : Type u_8
      b : AffineBasis ι R E
      x : E
      ⊢ Membership.mem (setOf fun x => Exists fun s => Exists fun w => And (∀ (i : ι …
    -/
  · rintro ⟨s, w, hw₀, hw₁, rfl⟩ i
    /-
      case h.refine_1.intro.intro.intro.intro
      R : Type u_1
      E : Type u_3
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      ι : Type u_8
      b : AffineBasis ι R E
      s : Finset ι
      w : ι → R
      hw₀ : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
      hw₁ : Eq (s.sum w) 1
      i : ι
      ⊢ LE.le 0 ((b.coord i) ((Finset.affineCombination R s ⇑b) w))
    -/
    by_cases hi : i ∈ s
      /-
        case pos
        R : Type u_1
        E : Type u_3
        inst✝² : LinearOrderedField R
        inst✝¹ : AddCommGroup E
        inst✝ : Module R E
        ι : Type u_8
        b : AffineBasis ι R E
        s : Finset ι
        w : ι → R
        hw₀ : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
        hw₁ : Eq (s.sum w) 1
        i : ι
        hi : Membership.mem s i
        ⊢ LE.le 0 ((b.coord i) ((Finset.affineCombination R s ⇑b) w))
      -/
    · rw [b.coord_apply_combination_of_mem hi hw₁]
      /-
        case pos
        R : Type u_1
        E : Type u_3
        inst✝² : LinearOrderedField R
        inst✝¹ : AddCommGroup E
        inst✝ : Module R E
        ι : Type u_8
        b : AffineBasis ι R E
        s : Finset ι
        w : ι → R
        hw₀ : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
        hw₁ : Eq (s.sum w) 1
        i : ι
        hi : Membership.mem s i
        ⊢ LE.le 0 (w i)
      -/
      exact hw₀ i hi
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        E : Type u_3
        inst✝² : LinearOrderedField R
        inst✝¹ : AddCommGroup E
        inst✝ : Module R E
        ι : Type u_8
        b : AffineBasis ι R E
        s : Finset ι
        w : ι → R
        hw₀ : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
        hw₁ : Eq (s.sum w) 1
        i : ι
        hi : Not (Membership.mem s i)
        ⊢ LE.le 0 ((b.coord i) ((Finset.affineCombination R s ⇑b) w))
      -/
    · rw [b.coord_apply_combination_of_not_mem hi hw₁]
      /-
        🎉 no goals
      -/
  · have hx' : x ∈ affineSpan R (range b) := by
      rw [b.tot]
      exact AffineSubspace.mem_top R E x
    /-
      case h.refine_2
      R : Type u_1
      E : Type u_3
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      ι : Type u_8
      b : AffineBasis ι R E
      x : E
      hx : Membership.mem (setOf fun x => ∀ (i : ι), LE.le 0 ((b.coord i) x)) x
      hx' : Membership.mem (affineSpan R (Set.range ⇑b)) x
      ⊢ Membership.mem (setOf fun x => Exists fun s => Exists fun w => And (∀ (i : ι …
    -/
    obtain ⟨s, w, hw₁, rfl⟩ := (mem_affineSpan_iff_eq_affineCombination R E).mp hx'
    /-
      case h.refine_2.intro.intro.intro
      R : Type u_1
      E : Type u_3
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      ι : Type u_8
      b : AffineBasis ι R E
      s : Finset ι
      w : ι → R
      hw₁ : Eq (s.sum fun i => w i) 1
      hx : Membership.mem (setOf fun x => ∀ (i : ι), LE.le 0 ((b.coord i) x)) ((Fins …
      hx' : Membership.mem (affineSpan R (Set.range ⇑b)) ((Finset.affineCombination  …
      ⊢ Membership.mem (setOf fun x => Exists fun s => Exists fun w => And (∀ (i : ι …
    -/
    refine ⟨s, w, ?_, hw₁, rfl⟩
    /-
      case h.refine_2.intro.intro.intro
      R : Type u_1
      E : Type u_3
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      ι : Type u_8
      b : AffineBasis ι R E
      s : Finset ι
      w : ι → R
      hw₁ : Eq (s.sum fun i => w i) 1
      hx : Membership.mem (setOf fun x => ∀ (i : ι), LE.le 0 ((b.coord i) x)) ((Fins …
      hx' : Membership.mem (affineSpan R (Set.range ⇑b)) ((Finset.affineCombination  …
      ⊢ ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
    -/
    intro i hi
    /-
      case h.refine_2.intro.intro.intro
      R : Type u_1
      E : Type u_3
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      ι : Type u_8
      b : AffineBasis ι R E
      s : Finset ι
      w : ι → R
      hw₁ : Eq (s.sum fun i => w i) 1
      hx : Membership.mem (setOf fun x => ∀ (i : ι), LE.le 0 ((b.coord i) x)) ((Fins …
      hx' : Membership.mem (affineSpan R (Set.range ⇑b)) ((Finset.affineCombination  …
      i : ι
      hi : Membership.mem s i
      ⊢ LE.le 0 (w i)
    -/
    specialize hx i
    /-
      case h.refine_2.intro.intro.intro
      R : Type u_1
      E : Type u_3
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      ι : Type u_8
      b : AffineBasis ι R E
      s : Finset ι
      w : ι → R
      hw₁ : Eq (s.sum fun i => w i) 1
      hx' : Membership.mem (affineSpan R (Set.range ⇑b)) ((Finset.affineCombination  …
      i : ι
      hi : Membership.mem s i
      hx : LE.le 0 ((b.coord i) ((Finset.affineCombination R s ⇑b) w))
      ⊢ LE.le 0 (w i)
    -/
    rw [b.coord_apply_combination_of_mem hi hw₁] at hx
    /-
      case h.refine_2.intro.intro.intro
      R : Type u_1
      E : Type u_3
      inst✝² : LinearOrderedField R
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      ι : Type u_8
      b : AffineBasis ι R E
      s : Finset ι
      w : ι → R
      hw₁ : Eq (s.sum fun i => w i) 1
      hx' : Membership.mem (affineSpan R (Set.range ⇑b)) ((Finset.affineCombination  …
      i : ι
      hi : Membership.mem s i
      hx : LE.le 0 (w i)
      ⊢ LE.le 0 (w i)
    -/
    exact hx
    /-
      🎉 no goals
    -/


/-- Two simplices glue nicely if the union of their vertices is affine independent. -/
lemma AffineIndependent.convexHull_inter (hs : AffineIndependent R ((↑) : s → E))
    (ht₁ : t₁ ⊆ s) (ht₂ : t₂ ⊆ s) :
    convexHull R (t₁ ∩ t₂ : Set E) = convexHull R t₁ ∩ convexHull R t₂ := by
  classical
  refine (Set.subset_inter (convexHull_mono inf_le_left) <|
    convexHull_mono inf_le_right).antisymm ?_
  simp_rw [Set.subset_def, mem_inter_iff, Set.inf_eq_inter, ← coe_inter, mem_convexHull']
  rintro x ⟨⟨w₁, h₁w₁, h₂w₁, h₃w₁⟩, w₂, -, h₂w₂, h₃w₂⟩
  let w (x : E) : R := (if x ∈ t₁ then w₁ x else 0) - if x ∈ t₂ then w₂ x else 0
  have h₁w : ∑ i ∈ s, w i = 0 := by simp [w, Finset.inter_eq_right.2, *]
  replace hs := hs.eq_zero_of_sum_eq_zero_subtype h₁w <| by
    simp only [w, sub_smul, zero_smul, ite_smul, Finset.sum_sub_distrib, ← Finset.sum_filter, h₃w₁,
      Finset.filter_mem_eq_inter, Finset.inter_eq_right.2 ht₁, Finset.inter_eq_right.2 ht₂, h₃w₂,
      sub_self]
  have ht (x) (hx₁ : x ∈ t₁) (hx₂ : x ∉ t₂) : w₁ x = 0 := by
    simpa [w, hx₁, hx₂] using hs _ (ht₁ hx₁)
  refine ⟨w₁, ?_, ?_, ?_⟩
  · simp only [and_imp, Finset.mem_inter]
    exact fun y hy₁ _ ↦ h₁w₁ y hy₁
  all_goals
  · rwa [sum_subset inter_subset_left]
    rintro x
    simp_intro hx₁ hx₂
    simp [ht x hx₁ hx₂]


open scoped Classical in
/-- Two simplices glue nicely if the union of their vertices is affine independent.

Note that `AffineIndependent.convexHull_inter` should be more versatile in most use cases. -/
lemma AffineIndependent.convexHull_inter' (hs : AffineIndependent R ((↑) : ↑(t₁ ∪ t₂) → E)) :
    convexHull R (t₁ ∩ t₂ : Set E) = convexHull R t₁ ∩ convexHull R t₂ :=
  hs.convexHull_inter subset_union_left subset_union_right


lemma mem_convexHull_pi (h : ∀ i ∈ s, x i ∈ convexHull 𝕜 (t i)) : x ∈ convexHull 𝕜 (s.pi t) := by
  classical
  cases nonempty_fintype ι
  wlog hs : s = Set.univ generalizing s t
  · rw [← pi_univ_ite]
    refine this (fun i _ ↦ ?_) rfl
    split_ifs with hi
    · exact h i hi
    · simp
  subst hs
  simp only [Set.mem_univ, mem_convexHull_iff_exists_fintype, true_implies, Set.mem_pi] at h
  choose κ _ w f hw₀ hw₁ hft hf using h
  refine mem_convexHull_of_exists_fintype (fun k : Π i, κ i ↦ ∏ i, w i (k i)) (fun g i ↦ f _ (g i))
    (fun g ↦ prod_nonneg fun _ _ ↦ hw₀ _ _) ?_ (fun _ _ _ ↦ hft _ _) ?_
  · rw [← Fintype.prod_sum]
    exact prod_eq_one fun _ _ ↦ hw₁ _
  ext i
  calc
    _ = ∑ g : ∀ i, κ i, (∏ i, w i (g i)) • f i (g i) := by
      simp only [Finset.sum_apply, Pi.smul_apply]
    _ = ∑ j : κ i, ∑ g : {g : ∀ k, κ k // g i = j},
          (∏ k, w k (g.1 k)) • f i ((g : ∀ i, κ i) i) := by
      rw [← Fintype.sum_fiberwise fun g : ∀ k, κ k ↦ g i]
    _ = ∑ j : κ i, (∑ g : {g : ∀ k, κ k // g i = j}, ∏ k, w k (g.1 k)) • f i j := by
      simp_rw [sum_smul]
      congr! with j _ g _
      exact g.2
    _ = ∑ j : κ i, w i j • f i j := ?_
    _ = x i := hf _
  congr! with j _
  calc
    ∑ g : {g : ∀ k, κ k // g i = j}, ∏ k, w k (g.1 k)
      = ∑ g ∈ piFinset fun k ↦ if hk : k = i then hk ▸ {j} else univ, ∏ k, w k (g k) :=
      Finset.sum_bij' (fun g _ ↦ g) (fun g hg ↦ ⟨g, by simpa using mem_piFinset.1 hg i⟩)
        (by aesop) (by simp) (by simp) (by simp) (by simp)
    _ = w i j := by
      rw [← prod_univ_sum, ← prod_mul_prod_compl, Finset.prod_singleton, Finset.sum_eq_single,
        Finset.prod_eq_one, mul_one] <;> simp +contextual [hw₁]


@[simp] lemma convexHull_pi (s : Set ι) (t : Π i, Set (E i)) :
    convexHull 𝕜 (s.pi t) = s.pi (fun i ↦ convexHull 𝕜 (t i)) :=
  Set.Subset.antisymm (convexHull_min (Set.pi_mono fun _ _ ↦ subset_convexHull _ _) <| convex_pi <|
    fun _ _ ↦ convex_convexHull _ _) fun _ ↦ mem_convexHull_pi


