theorem lineMap_mono_left (ha : a ≤ a') (hr : r ≤ 1) : lineMap a b r ≤ lineMap a' b r := by
  /-
    k : Type u_1
    E : Type u_2
    inst✝³ : OrderedRing k
    inst✝² : OrderedAddCommGroup E
    inst✝¹ : Module k E
    inst✝ : OrderedSMul k E
    a a' b : E
    r : k
    ha : LE.le a a'
    hr : LE.le r 1
    ⊢ LE.le ((AffineMap.lineMap a b) r) ((AffineMap.lineMap a' b) r)
  -/
  simp only [lineMap_apply_module]
  /-
    k : Type u_1
    E : Type u_2
    inst✝³ : OrderedRing k
    inst✝² : OrderedAddCommGroup E
    inst✝¹ : Module k E
    inst✝ : OrderedSMul k E
    a a' b : E
    r : k
    ha : LE.le a a'
    hr : LE.le r 1
    ⊢ LE.le (HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 r) a) (HSMul.hSMul r b)) (HAdd.hA …
  -/
  exact add_le_add_right (smul_le_smul_of_nonneg_left ha (sub_nonneg.2 hr)) _
  /-
    🎉 no goals
  -/


theorem lineMap_strict_mono_left (ha : a < a') (hr : r < 1) : lineMap a b r < lineMap a' b r := by
  /-
    k : Type u_1
    E : Type u_2
    inst✝³ : OrderedRing k
    inst✝² : OrderedAddCommGroup E
    inst✝¹ : Module k E
    inst✝ : OrderedSMul k E
    a a' b : E
    r : k
    ha : LT.lt a a'
    hr : LT.lt r 1
    ⊢ LT.lt ((AffineMap.lineMap a b) r) ((AffineMap.lineMap a' b) r)
  -/
  simp only [lineMap_apply_module]
  /-
    k : Type u_1
    E : Type u_2
    inst✝³ : OrderedRing k
    inst✝² : OrderedAddCommGroup E
    inst✝¹ : Module k E
    inst✝ : OrderedSMul k E
    a a' b : E
    r : k
    ha : LT.lt a a'
    hr : LT.lt r 1
    ⊢ LT.lt (HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 r) a) (HSMul.hSMul r b)) (HAdd.hA …
  -/
  exact add_lt_add_right (smul_lt_smul_of_pos_left ha (sub_pos.2 hr)) _
  /-
    🎉 no goals
  -/


theorem lineMap_mono_right (hb : b ≤ b') (hr : 0 ≤ r) : lineMap a b r ≤ lineMap a b' r := by
  /-
    k : Type u_1
    E : Type u_2
    inst✝³ : OrderedRing k
    inst✝² : OrderedAddCommGroup E
    inst✝¹ : Module k E
    inst✝ : OrderedSMul k E
    a b b' : E
    r : k
    hb : LE.le b b'
    hr : LE.le 0 r
    ⊢ LE.le ((AffineMap.lineMap a b) r) ((AffineMap.lineMap a b') r)
  -/
  simp only [lineMap_apply_module]
  /-
    k : Type u_1
    E : Type u_2
    inst✝³ : OrderedRing k
    inst✝² : OrderedAddCommGroup E
    inst✝¹ : Module k E
    inst✝ : OrderedSMul k E
    a b b' : E
    r : k
    hb : LE.le b b'
    hr : LE.le 0 r
    ⊢ LE.le (HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 r) a) (HSMul.hSMul r b)) (HAdd.hA …
  -/
  exact add_le_add_left (smul_le_smul_of_nonneg_left hb hr) _
  /-
    🎉 no goals
  -/


theorem lineMap_strict_mono_right (hb : b < b') (hr : 0 < r) : lineMap a b r < lineMap a b' r := by
  /-
    k : Type u_1
    E : Type u_2
    inst✝³ : OrderedRing k
    inst✝² : OrderedAddCommGroup E
    inst✝¹ : Module k E
    inst✝ : OrderedSMul k E
    a b b' : E
    r : k
    hb : LT.lt b b'
    hr : LT.lt 0 r
    ⊢ LT.lt ((AffineMap.lineMap a b) r) ((AffineMap.lineMap a b') r)
  -/
  simp only [lineMap_apply_module]
  /-
    k : Type u_1
    E : Type u_2
    inst✝³ : OrderedRing k
    inst✝² : OrderedAddCommGroup E
    inst✝¹ : Module k E
    inst✝ : OrderedSMul k E
    a b b' : E
    r : k
    hb : LT.lt b b'
    hr : LT.lt 0 r
    ⊢ LT.lt (HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 r) a) (HSMul.hSMul r b)) (HAdd.hA …
  -/
  exact add_lt_add_left (smul_lt_smul_of_pos_left hb hr) _
  /-
    🎉 no goals
  -/


theorem lineMap_mono_endpoints (ha : a ≤ a') (hb : b ≤ b') (h₀ : 0 ≤ r) (h₁ : r ≤ 1) :
    lineMap a b r ≤ lineMap a' b' r :=
  (lineMap_mono_left ha h₁).trans (lineMap_mono_right hb h₀)


theorem lineMap_strict_mono_endpoints (ha : a < a') (hb : b < b') (h₀ : 0 ≤ r) (h₁ : r ≤ 1) :
    lineMap a b r < lineMap a' b' r := by
  /-
    k : Type u_1
    E : Type u_2
    inst✝³ : OrderedRing k
    inst✝² : OrderedAddCommGroup E
    inst✝¹ : Module k E
    inst✝ : OrderedSMul k E
    a a' b b' : E
    r : k
    ha : LT.lt a a'
    hb : LT.lt b b'
    h₀ : LE.le 0 r
    h₁ : LE.le r 1
    ⊢ LT.lt ((AffineMap.lineMap a b) r) ((AffineMap.lineMap a' b') r)
  -/
  rcases h₀.eq_or_lt with (rfl | h₀); · simpa
                                        /-
                                          🎉 no goals
                                        -/
  /-
    case inr
    k : Type u_1
    E : Type u_2
    inst✝³ : OrderedRing k
    inst✝² : OrderedAddCommGroup E
    inst✝¹ : Module k E
    inst✝ : OrderedSMul k E
    a a' b b' : E
    r : k
    ha : LT.lt a a'
    hb : LT.lt b b'
    h₀✝ : LE.le 0 r
    h₁ : LE.le r 1
    h₀ : LT.lt 0 r
    ⊢ LT.lt ((AffineMap.lineMap a b) r) ((AffineMap.lineMap a' b') r)
  -/
  exact (lineMap_mono_left ha.le h₁).trans_lt (lineMap_strict_mono_right hb h₀)
  /-
    🎉 no goals
  -/


theorem lineMap_lt_lineMap_iff_of_lt (h : r < r') : lineMap a b r < lineMap a b r' ↔ a < b := by
  /-
    k : Type u_1
    E : Type u_2
    inst✝³ : OrderedRing k
    inst✝² : OrderedAddCommGroup E
    inst✝¹ : Module k E
    inst✝ : OrderedSMul k E
    a b : E
    r r' : k
    h : LT.lt r r'
    ⊢ Iff (LT.lt ((AffineMap.lineMap a b) r) ((AffineMap.lineMap a b) r')) (LT.lt  …
  -/
  simp only [lineMap_apply_module]
  rw [← lt_sub_iff_add_lt, add_sub_assoc, ← sub_lt_iff_lt_add', ← sub_smul, ← sub_smul,
    sub_sub_sub_cancel_left, smul_lt_smul_iff_of_pos_left (sub_pos.2 h)]


theorem left_lt_lineMap_iff_lt (h : 0 < r) : a < lineMap a b r ↔ a < b :=
                /-
                  k : Type u_1
                  E : Type u_2
                  inst✝³ : OrderedRing k
                  inst✝² : OrderedAddCommGroup E
                  inst✝¹ : Module k E
                  inst✝ : OrderedSMul k E
                  a b : E
                  r : k
                  h : LT.lt 0 r
                  ⊢ Iff (LT.lt a ((AffineMap.lineMap a b) r)) (LT.lt ((AffineMap.lineMap a b) 0) …
                -/
  Iff.trans (by rw [lineMap_apply_zero]) (lineMap_lt_lineMap_iff_of_lt h)
                /-
                  🎉 no goals
                -/


theorem lineMap_lt_left_iff_lt (h : 0 < r) : lineMap a b r < a ↔ b < a :=
  left_lt_lineMap_iff_lt (E := Eᵒᵈ) h


theorem lineMap_lt_right_iff_lt (h : r < 1) : lineMap a b r < b ↔ a < b :=
                /-
                  k : Type u_1
                  E : Type u_2
                  inst✝³ : OrderedRing k
                  inst✝² : OrderedAddCommGroup E
                  inst✝¹ : Module k E
                  inst✝ : OrderedSMul k E
                  a b : E
                  r : k
                  h : LT.lt r 1
                  ⊢ Iff (LT.lt ((AffineMap.lineMap a b) r) b) (LT.lt ((AffineMap.lineMap a b) r) …
                -/
  Iff.trans (by rw [lineMap_apply_one]) (lineMap_lt_lineMap_iff_of_lt h)
                /-
                  🎉 no goals
                -/


theorem right_lt_lineMap_iff_lt (h : r < 1) : b < lineMap a b r ↔ b < a :=
  lineMap_lt_right_iff_lt (E := Eᵒᵈ) h


theorem midpoint_le_midpoint (ha : a ≤ a') (hb : b ≤ b') : midpoint k a b ≤ midpoint k a' b' :=
  lineMap_mono_endpoints ha hb (invOf_nonneg.2 zero_le_two) <| invOf_le_one one_le_two


theorem lineMap_le_lineMap_iff_of_lt (h : r < r') : lineMap a b r ≤ lineMap a b r' ↔ a ≤ b := by
  /-
    k : Type u_1
    E : Type u_2
    inst✝³ : LinearOrderedField k
    inst✝² : OrderedAddCommGroup E
    inst✝¹ : Module k E
    inst✝ : OrderedSMul k E
    a b : E
    r r' : k
    h : LT.lt r r'
    ⊢ Iff (LE.le ((AffineMap.lineMap a b) r) ((AffineMap.lineMap a b) r')) (LE.le  …
  -/
  simp only [lineMap_apply_module]
  rw [← le_sub_iff_add_le, add_sub_assoc, ← sub_le_iff_le_add', ← sub_smul, ← sub_smul,
    sub_sub_sub_cancel_left, smul_le_smul_iff_of_pos_left (sub_pos.2 h)]


theorem left_le_lineMap_iff_le (h : 0 < r) : a ≤ lineMap a b r ↔ a ≤ b :=
                /-
                  k : Type u_1
                  E : Type u_2
                  inst✝³ : LinearOrderedField k
                  inst✝² : OrderedAddCommGroup E
                  inst✝¹ : Module k E
                  inst✝ : OrderedSMul k E
                  a b : E
                  r : k
                  h : LT.lt 0 r
                  ⊢ Iff (LE.le a ((AffineMap.lineMap a b) r)) (LE.le ((AffineMap.lineMap a b) 0) …
                -/
  Iff.trans (by rw [lineMap_apply_zero]) (lineMap_le_lineMap_iff_of_lt h)
                /-
                  🎉 no goals
                -/


@[simp]
theorem left_le_midpoint : a ≤ midpoint k a b ↔ a ≤ b :=
  left_le_lineMap_iff_le <| inv_pos.2 zero_lt_two


theorem lineMap_le_left_iff_le (h : 0 < r) : lineMap a b r ≤ a ↔ b ≤ a :=
  left_le_lineMap_iff_le (E := Eᵒᵈ) h


@[simp]
theorem midpoint_le_left : midpoint k a b ≤ a ↔ b ≤ a :=
  lineMap_le_left_iff_le <| inv_pos.2 zero_lt_two


theorem lineMap_le_right_iff_le (h : r < 1) : lineMap a b r ≤ b ↔ a ≤ b :=
                /-
                  k : Type u_1
                  E : Type u_2
                  inst✝³ : LinearOrderedField k
                  inst✝² : OrderedAddCommGroup E
                  inst✝¹ : Module k E
                  inst✝ : OrderedSMul k E
                  a b : E
                  r : k
                  h : LT.lt r 1
                  ⊢ Iff (LE.le ((AffineMap.lineMap a b) r) b) (LE.le ((AffineMap.lineMap a b) r) …
                -/
  Iff.trans (by rw [lineMap_apply_one]) (lineMap_le_lineMap_iff_of_lt h)
                /-
                  🎉 no goals
                -/


@[simp]
theorem midpoint_le_right : midpoint k a b ≤ b ↔ a ≤ b := lineMap_le_right_iff_le two_inv_lt_one


theorem right_le_lineMap_iff_le (h : r < 1) : b ≤ lineMap a b r ↔ b ≤ a :=
  lineMap_le_right_iff_le (E := Eᵒᵈ) h


@[simp]
theorem right_le_midpoint : b ≤ midpoint k a b ↔ b ≤ a := right_le_lineMap_iff_le two_inv_lt_one


local notation "c" => lineMap a b r


/-- Given `c = lineMap a b r`, `a < c`, the point `(c, f c)` is non-strictly below the
segment `[(a, f a), (b, f b)]` if and only if `slope f a c ≤ slope f a b`. -/
theorem map_le_lineMap_iff_slope_le_slope_left (h : 0 < r * (b - a)) :
    f c ≤ lineMap (f a) (f b) r ↔ slope f a c ≤ slope f a b := by
  rw [lineMap_apply, lineMap_apply, slope, slope, vsub_eq_sub, vsub_eq_sub, vsub_eq_sub,
    vadd_eq_add, vadd_eq_add, smul_eq_mul, add_sub_cancel_right, smul_sub, smul_sub, smul_sub,
    sub_le_iff_le_add, mul_inv_rev, mul_smul, mul_smul, ← smul_sub, ← smul_sub, ← smul_add,
    smul_smul, ← mul_inv_rev, inv_smul_le_iff_of_pos h, smul_smul,
    mul_inv_cancel_right₀ (right_ne_zero_of_mul h.ne'), smul_add,
    smul_inv_smul₀ (left_ne_zero_of_mul h.ne')]


/-- Given `c = lineMap a b r`, `a < c`, the point `(c, f c)` is non-strictly above the
segment `[(a, f a), (b, f b)]` if and only if `slope f a b ≤ slope f a c`. -/
theorem lineMap_le_map_iff_slope_le_slope_left (h : 0 < r * (b - a)) :
    lineMap (f a) (f b) r ≤ f c ↔ slope f a b ≤ slope f a c :=
  map_le_lineMap_iff_slope_le_slope_left (E := Eᵒᵈ) (f := f) (a := a) (b := b) (r := r) h


/-- Given `c = lineMap a b r`, `a < c`, the point `(c, f c)` is strictly below the
segment `[(a, f a), (b, f b)]` if and only if `slope f a c < slope f a b`. -/
theorem map_lt_lineMap_iff_slope_lt_slope_left (h : 0 < r * (b - a)) :
    f c < lineMap (f a) (f b) r ↔ slope f a c < slope f a b :=
  lt_iff_lt_of_le_iff_le' (lineMap_le_map_iff_slope_le_slope_left h)
    (map_le_lineMap_iff_slope_le_slope_left h)


/-- Given `c = lineMap a b r`, `a < c`, the point `(c, f c)` is strictly above the
segment `[(a, f a), (b, f b)]` if and only if `slope f a b < slope f a c`. -/
theorem lineMap_lt_map_iff_slope_lt_slope_left (h : 0 < r * (b - a)) :
    lineMap (f a) (f b) r < f c ↔ slope f a b < slope f a c :=
  map_lt_lineMap_iff_slope_lt_slope_left (E := Eᵒᵈ) (f := f) (a := a) (b := b) (r := r) h


/-- Given `c = lineMap a b r`, `c < b`, the point `(c, f c)` is non-strictly below the
segment `[(a, f a), (b, f b)]` if and only if `slope f a b ≤ slope f c b`. -/
theorem map_le_lineMap_iff_slope_le_slope_right (h : 0 < (1 - r) * (b - a)) :
    f c ≤ lineMap (f a) (f b) r ↔ slope f a b ≤ slope f c b := by
  /-
    k : Type u_1
    E : Type u_2
    inst✝³ : LinearOrderedField k
    inst✝² : OrderedAddCommGroup E
    inst✝¹ : Module k E
    inst✝ : OrderedSMul k E
    f : k → E
    a b r : k
    h : LT.lt 0 (HMul.hMul (HSub.hSub 1 r) (HSub.hSub b a))
    ⊢ Iff (LE.le (f ((AffineMap.lineMap a b) r)) ((AffineMap.lineMap (f a) (f b))  …
  -/
  rw [← lineMap_apply_one_sub, ← lineMap_apply_one_sub _ _ r]
  /-
    k : Type u_1
    E : Type u_2
    inst✝³ : LinearOrderedField k
    inst✝² : OrderedAddCommGroup E
    inst✝¹ : Module k E
    inst✝ : OrderedSMul k E
    f : k → E
    a b r : k
    h : LT.lt 0 (HMul.hMul (HSub.hSub 1 r) (HSub.hSub b a))
    ⊢ Iff (LE.le (f ((AffineMap.lineMap b a) (HSub.hSub 1 r))) ((AffineMap.lineMap …
  -/
  revert h; generalize 1 - r = r'; clear! r; intro h
  /-
    k : Type u_1
    E : Type u_2
    inst✝³ : LinearOrderedField k
    inst✝² : OrderedAddCommGroup E
    inst✝¹ : Module k E
    inst✝ : OrderedSMul k E
    f : k → E
    a b r' : k
    h : LT.lt 0 (HMul.hMul r' (HSub.hSub b a))
    ⊢ Iff (LE.le (f ((AffineMap.lineMap b a) r')) ((AffineMap.lineMap (f b) (f a)) …
  -/
  simp_rw [lineMap_apply, slope, vsub_eq_sub, vadd_eq_add, smul_eq_mul]
  rw [sub_add_eq_sub_sub_swap, sub_self, zero_sub, neg_mul_eq_mul_neg, neg_sub,
    le_inv_smul_iff_of_pos h, smul_smul, mul_inv_cancel_right₀, le_sub_comm, ← neg_sub (f b),
    smul_neg, neg_add_eq_sub]
    /-
      case h
      k : Type u_1
      E : Type u_2
      inst✝³ : LinearOrderedField k
      inst✝² : OrderedAddCommGroup E
      inst✝¹ : Module k E
      inst✝ : OrderedSMul k E
      f : k → E
      a b r' : k
      h : LT.lt 0 (HMul.hMul r' (HSub.hSub b a))
      ⊢ Ne (HSub.hSub b a) 0
    -/
  · exact right_ne_zero_of_mul h.ne'
    /-
      🎉 no goals
    -/


/-- Given `c = lineMap a b r`, `c < b`, the point `(c, f c)` is non-strictly above the
segment `[(a, f a), (b, f b)]` if and only if `slope f c b ≤ slope f a b`. -/
theorem lineMap_le_map_iff_slope_le_slope_right (h : 0 < (1 - r) * (b - a)) :
    lineMap (f a) (f b) r ≤ f c ↔ slope f c b ≤ slope f a b :=
  map_le_lineMap_iff_slope_le_slope_right (E := Eᵒᵈ) (f := f) (a := a) (b := b) (r := r) h


/-- Given `c = lineMap a b r`, `c < b`, the point `(c, f c)` is strictly below the
segment `[(a, f a), (b, f b)]` if and only if `slope f a b < slope f c b`. -/
theorem map_lt_lineMap_iff_slope_lt_slope_right (h : 0 < (1 - r) * (b - a)) :
    f c < lineMap (f a) (f b) r ↔ slope f a b < slope f c b :=
  lt_iff_lt_of_le_iff_le' (lineMap_le_map_iff_slope_le_slope_right h)
    (map_le_lineMap_iff_slope_le_slope_right h)


/-- Given `c = lineMap a b r`, `c < b`, the point `(c, f c)` is strictly above the
segment `[(a, f a), (b, f b)]` if and only if `slope f c b < slope f a b`. -/
theorem lineMap_lt_map_iff_slope_lt_slope_right (h : 0 < (1 - r) * (b - a)) :
    lineMap (f a) (f b) r < f c ↔ slope f c b < slope f a b :=
  map_lt_lineMap_iff_slope_lt_slope_right (E := Eᵒᵈ) (f := f) (a := a) (b := b) (r := r) h


/-- Given `c = lineMap a b r`, `a < c < b`, the point `(c, f c)` is non-strictly below the
segment `[(a, f a), (b, f b)]` if and only if `slope f a c ≤ slope f c b`. -/
theorem map_le_lineMap_iff_slope_le_slope (hab : a < b) (h₀ : 0 < r) (h₁ : r < 1) :
    f c ≤ lineMap (f a) (f b) r ↔ slope f a c ≤ slope f c b := by
  rw [map_le_lineMap_iff_slope_le_slope_left (mul_pos h₀ (sub_pos.2 hab)), ←
    lineMap_slope_lineMap_slope_lineMap f a b r, right_le_lineMap_iff_le h₁]


/-- Given `c = lineMap a b r`, `a < c < b`, the point `(c, f c)` is non-strictly above the
segment `[(a, f a), (b, f b)]` if and only if `slope f c b ≤ slope f a c`. -/
theorem lineMap_le_map_iff_slope_le_slope (hab : a < b) (h₀ : 0 < r) (h₁ : r < 1) :
    lineMap (f a) (f b) r ≤ f c ↔ slope f c b ≤ slope f a c :=
  map_le_lineMap_iff_slope_le_slope (E := Eᵒᵈ) hab h₀ h₁


/-- Given `c = lineMap a b r`, `a < c < b`, the point `(c, f c)` is strictly below the
segment `[(a, f a), (b, f b)]` if and only if `slope f a c < slope f c b`. -/
theorem map_lt_lineMap_iff_slope_lt_slope (hab : a < b) (h₀ : 0 < r) (h₁ : r < 1) :
    f c < lineMap (f a) (f b) r ↔ slope f a c < slope f c b :=
  lt_iff_lt_of_le_iff_le' (lineMap_le_map_iff_slope_le_slope hab h₀ h₁)
    (map_le_lineMap_iff_slope_le_slope hab h₀ h₁)


/-- Given `c = lineMap a b r`, `a < c < b`, the point `(c, f c)` is strictly above the
segment `[(a, f a), (b, f b)]` if and only if `slope f c b < slope f a c`. -/
theorem lineMap_lt_map_iff_slope_lt_slope (hab : a < b) (h₀ : 0 < r) (h₁ : r < 1) :
    lineMap (f a) (f b) r < f c ↔ slope f c b < slope f a c :=
  map_lt_lineMap_iff_slope_lt_slope (E := Eᵒᵈ) hab h₀ h₁


