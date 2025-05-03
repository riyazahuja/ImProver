/-- The Hamming distance function to the naturals. -/
def hammingDist (x y : ∀ i, β i) : ℕ := #{i | x i ≠ y i}


/-- Corresponds to `dist_self`. -/
@[simp]
theorem hammingDist_self (x : ∀ i, β i) : hammingDist x x = 0 := by
  /-
    ι : Type u_2
    β : ι → Type u_3
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → DecidableEq (β i)
    x : (i : ι) → β i
    ⊢ Eq (hammingDist x x) 0
  -/
  rw [hammingDist, card_eq_zero, filter_eq_empty_iff]
  /-
    ι : Type u_2
    β : ι → Type u_3
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → DecidableEq (β i)
    x : (i : ι) → β i
    ⊢ ∀ ⦃x_1 : ι⦄, Membership.mem Finset.univ x_1 → Not (Ne (x x_1) (x x_1))
  -/
  exact fun _ _ H => H rfl
  /-
    🎉 no goals
  -/


/-- Corresponds to `dist_nonneg`. -/
theorem hammingDist_nonneg {x y : ∀ i, β i} : 0 ≤ hammingDist x y :=
  zero_le _


/-- Corresponds to `dist_comm`. -/
theorem hammingDist_comm (x y : ∀ i, β i) : hammingDist x y = hammingDist y x := by
  /-
    ι : Type u_2
    β : ι → Type u_3
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → DecidableEq (β i)
    x y : (i : ι) → β i
    ⊢ Eq (hammingDist x y) (hammingDist y x)
  -/
  simp_rw [hammingDist, ne_comm]
  /-
    🎉 no goals
  -/


/-- Corresponds to `dist_triangle`. -/
theorem hammingDist_triangle (x y z : ∀ i, β i) :
    hammingDist x z ≤ hammingDist x y + hammingDist y z := by
  classical
    unfold hammingDist
    refine le_trans (card_mono ?_) (card_union_le _ _)
    rw [← filter_or]
    exact monotone_filter_right _ fun i h ↦ (h.ne_or_ne _).imp_right Ne.symm


/-- Corresponds to `dist_triangle_left`. -/
theorem hammingDist_triangle_left (x y z : ∀ i, β i) :
    hammingDist x y ≤ hammingDist z x + hammingDist z y := by
  /-
    ι : Type u_2
    β : ι → Type u_3
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → DecidableEq (β i)
    x y z : (i : ι) → β i
    ⊢ LE.le (hammingDist x y) (HAdd.hAdd (hammingDist z x) (hammingDist z y))
  -/
  rw [hammingDist_comm z]
  /-
    ι : Type u_2
    β : ι → Type u_3
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → DecidableEq (β i)
    x y z : (i : ι) → β i
    ⊢ LE.le (hammingDist x y) (HAdd.hAdd (hammingDist x z) (hammingDist z y))
  -/
  exact hammingDist_triangle _ _ _
  /-
    🎉 no goals
  -/


/-- Corresponds to `dist_triangle_right`. -/
theorem hammingDist_triangle_right (x y z : ∀ i, β i) :
    hammingDist x y ≤ hammingDist x z + hammingDist y z := by
  /-
    ι : Type u_2
    β : ι → Type u_3
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → DecidableEq (β i)
    x y z : (i : ι) → β i
    ⊢ LE.le (hammingDist x y) (HAdd.hAdd (hammingDist x z) (hammingDist y z))
  -/
  rw [hammingDist_comm y]
  /-
    ι : Type u_2
    β : ι → Type u_3
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → DecidableEq (β i)
    x y z : (i : ι) → β i
    ⊢ LE.le (hammingDist x y) (HAdd.hAdd (hammingDist x z) (hammingDist z y))
  -/
  exact hammingDist_triangle _ _ _
  /-
    🎉 no goals
  -/


/-- Corresponds to `swap_dist`. -/
theorem swap_hammingDist : swap (@hammingDist _ β _ _) = hammingDist := by
  /-
    ι : Type u_2
    β : ι → Type u_3
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → DecidableEq (β i)
    ⊢ Eq (Function.swap hammingDist) hammingDist
  -/
  funext x y
  /-
    case h.h
    ι : Type u_2
    β : ι → Type u_3
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → DecidableEq (β i)
    x y : (i : ι) → β i
    ⊢ Eq (Function.swap hammingDist x y) (hammingDist x y)
  -/
  exact hammingDist_comm _ _
  /-
    🎉 no goals
  -/


/-- Corresponds to `eq_of_dist_eq_zero`. -/
theorem eq_of_hammingDist_eq_zero {x y : ∀ i, β i} : hammingDist x y = 0 → x = y := by
  simp_rw [hammingDist, card_eq_zero, filter_eq_empty_iff, Classical.not_not, funext_iff, mem_univ,
    forall_true_left, imp_self]


/-- Corresponds to `dist_eq_zero`. -/
@[simp]
theorem hammingDist_eq_zero {x y : ∀ i, β i} : hammingDist x y = 0 ↔ x = y :=
  ⟨eq_of_hammingDist_eq_zero, fun H => by
    /-
      ι : Type u_2
      β : ι → Type u_3
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → DecidableEq (β i)
      x y : (i : ι) → β i
      H : Eq x y
      ⊢ Eq (hammingDist x y) 0
    -/
    rw [H]
    /-
      ι : Type u_2
      β : ι → Type u_3
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → DecidableEq (β i)
      x y : (i : ι) → β i
      H : Eq x y
      ⊢ Eq (hammingDist y y) 0
    -/
    exact hammingDist_self _⟩
    /-
      🎉 no goals
    -/


/-- Corresponds to `zero_eq_dist`. -/
@[simp]
theorem hamming_zero_eq_dist {x y : ∀ i, β i} : 0 = hammingDist x y ↔ x = y := by
  /-
    ι : Type u_2
    β : ι → Type u_3
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → DecidableEq (β i)
    x y : (i : ι) → β i
    ⊢ Iff (Eq 0 (hammingDist x y)) (Eq x y)
  -/
  rw [eq_comm, hammingDist_eq_zero]
  /-
    🎉 no goals
  -/


/-- Corresponds to `dist_ne_zero`. -/
theorem hammingDist_ne_zero {x y : ∀ i, β i} : hammingDist x y ≠ 0 ↔ x ≠ y :=
  hammingDist_eq_zero.not


/-- Corresponds to `dist_pos`. -/
@[simp]
theorem hammingDist_pos {x y : ∀ i, β i} : 0 < hammingDist x y ↔ x ≠ y := by
  /-
    ι : Type u_2
    β : ι → Type u_3
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → DecidableEq (β i)
    x y : (i : ι) → β i
    ⊢ Iff (LT.lt 0 (hammingDist x y)) (Ne x y)
  -/
  rw [← hammingDist_ne_zero, iff_not_comm, not_lt, Nat.le_zero]
  /-
    🎉 no goals
  -/


theorem hammingDist_lt_one {x y : ∀ i, β i} : hammingDist x y < 1 ↔ x = y := by
  /-
    ι : Type u_2
    β : ι → Type u_3
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → DecidableEq (β i)
    x y : (i : ι) → β i
    ⊢ Iff (LT.lt (hammingDist x y) 1) (Eq x y)
  -/
  rw [Nat.lt_one_iff, hammingDist_eq_zero]
  /-
    🎉 no goals
  -/


theorem hammingDist_le_card_fintype {x y : ∀ i, β i} : hammingDist x y ≤ Fintype.card ι :=
  card_le_univ _


theorem hammingDist_comp_le_hammingDist (f : ∀ i, γ i → β i) {x y : ∀ i, γ i} :
    (hammingDist (fun i => f i (x i)) fun i => f i (y i)) ≤ hammingDist x y :=
  card_mono (monotone_filter_right _ fun i H1 H2 => H1 <| congr_arg (f i) H2)


theorem hammingDist_comp (f : ∀ i, γ i → β i) {x y : ∀ i, γ i} (hf : ∀ i, Injective (f i)) :
    (hammingDist (fun i => f i (x i)) fun i => f i (y i)) = hammingDist x y :=
  le_antisymm (hammingDist_comp_le_hammingDist _) <|
    card_mono (monotone_filter_right _ fun i H1 H2 => H1 <| hf i H2)


theorem hammingDist_smul_le_hammingDist [∀ i, SMul α (β i)] {k : α} {x y : ∀ i, β i} :
    hammingDist (k • x) (k • y) ≤ hammingDist x y :=
  hammingDist_comp_le_hammingDist fun i => (k • · : β i → β i)


/-- Corresponds to `dist_smul` with the discrete norm on `α`. -/
theorem hammingDist_smul [∀ i, SMul α (β i)] {k : α} {x y : ∀ i, β i}
    (hk : ∀ i, IsSMulRegular (β i) k) : hammingDist (k • x) (k • y) = hammingDist x y :=
  hammingDist_comp (fun i => (k • · : β i → β i)) hk


/-- The Hamming weight function to the naturals. -/
def hammingNorm (x : ∀ i, β i) : ℕ := #{i | x i ≠ 0}


/-- Corresponds to `dist_zero_right`. -/
@[simp]
theorem hammingDist_zero_right (x : ∀ i, β i) : hammingDist x 0 = hammingNorm x :=
  rfl


/-- Corresponds to `dist_zero_left`. -/
@[simp]
theorem hammingDist_zero_left : hammingDist (0 : ∀ i, β i) = hammingNorm :=
                     /-
                       ι : Type u_2
                       β : ι → Type u_3
                       inst✝² : Fintype ι
                       inst✝¹ : (i : ι) → DecidableEq (β i)
                       inst✝ : (i : ι) → Zero (β i)
                       x : (i : ι) → β i
                       ⊢ Eq (hammingDist 0 x) (hammingNorm x)
                     -/
  funext fun x => by rw [hammingDist_comm, hammingDist_zero_right]
                     /-
                       🎉 no goals
                     -/


/-- Corresponds to `norm_nonneg`. -/
theorem hammingNorm_nonneg {x : ∀ i, β i} : 0 ≤ hammingNorm x :=
  zero_le _


/-- Corresponds to `norm_zero`. -/
@[simp]
theorem hammingNorm_zero : hammingNorm (0 : ∀ i, β i) = 0 :=
  hammingDist_self _


/-- Corresponds to `norm_eq_zero`. -/
@[simp]
theorem hammingNorm_eq_zero {x : ∀ i, β i} : hammingNorm x = 0 ↔ x = 0 :=
  hammingDist_eq_zero


/-- Corresponds to `norm_ne_zero_iff`. -/
theorem hammingNorm_ne_zero_iff {x : ∀ i, β i} : hammingNorm x ≠ 0 ↔ x ≠ 0 :=
  hammingNorm_eq_zero.not


/-- Corresponds to `norm_pos_iff`. -/
@[simp]
theorem hammingNorm_pos_iff {x : ∀ i, β i} : 0 < hammingNorm x ↔ x ≠ 0 :=
  hammingDist_pos


theorem hammingNorm_lt_one {x : ∀ i, β i} : hammingNorm x < 1 ↔ x = 0 :=
  hammingDist_lt_one


theorem hammingNorm_le_card_fintype {x : ∀ i, β i} : hammingNorm x ≤ Fintype.card ι :=
  hammingDist_le_card_fintype


theorem hammingNorm_comp_le_hammingNorm (f : ∀ i, γ i → β i) {x : ∀ i, γ i} (hf : ∀ i, f i 0 = 0) :
    (hammingNorm fun i => f i (x i)) ≤ hammingNorm x := by
  /-
    ι : Type u_2
    β : ι → Type u_3
    inst✝⁴ : Fintype ι
    inst✝³ : (i : ι) → DecidableEq (β i)
    γ : ι → Type u_4
    inst✝² : (i : ι) → DecidableEq (γ i)
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : (i : ι) → Zero (γ i)
    f : (i : ι) → γ i → β i
    x : (i : ι) → γ i
    hf : ∀ (i : ι), Eq (f i 0) 0
    ⊢ LE.le (hammingNorm fun i => f i (x i)) (hammingNorm x)
  -/
  simpa only [← hammingDist_zero_right, hf] using hammingDist_comp_le_hammingDist f (y := fun _ ↦ 0)
  /-
    🎉 no goals
  -/


theorem hammingNorm_comp (f : ∀ i, γ i → β i) {x : ∀ i, γ i} (hf₁ : ∀ i, Injective (f i))
    (hf₂ : ∀ i, f i 0 = 0) : (hammingNorm fun i => f i (x i)) = hammingNorm x := by
  /-
    ι : Type u_2
    β : ι → Type u_3
    inst✝⁴ : Fintype ι
    inst✝³ : (i : ι) → DecidableEq (β i)
    γ : ι → Type u_4
    inst✝² : (i : ι) → DecidableEq (γ i)
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : (i : ι) → Zero (γ i)
    f : (i : ι) → γ i → β i
    x : (i : ι) → γ i
    hf₁ : ∀ (i : ι), Function.Injective (f i)
    hf₂ : ∀ (i : ι), Eq (f i 0) 0
    ⊢ Eq (hammingNorm fun i => f i (x i)) (hammingNorm x)
  -/
  simpa only [← hammingDist_zero_right, hf₂] using hammingDist_comp f hf₁ (y := fun _ ↦ 0)
  /-
    🎉 no goals
  -/


theorem hammingNorm_smul_le_hammingNorm [Zero α] [∀ i, SMulWithZero α (β i)] {k : α}
    {x : ∀ i, β i} : hammingNorm (k • x) ≤ hammingNorm x :=
                                                                         /-
                                                                           α : Type u_1
                                                                           ι : Type u_2
                                                                           β : ι → Type u_3
                                                                           inst✝⁴ : Fintype ι
                                                                           inst✝³ : (i : ι) → DecidableEq (β i)
                                                                           inst✝² : (i : ι) → Zero (β i)
                                                                           inst✝¹ : Zero α
                                                                           inst✝ : (i : ι) → SMulWithZero α (β i)
                                                                           k : α
                                                                           x : (i : ι) → β i
                                                                           i : ι
                                                                           ⊢ Eq ((fun i c => HSMul.hSMul k c) i 0) 0
                                                                         -/
  hammingNorm_comp_le_hammingNorm (fun i (c : β i) => k • c) fun i => by simp_rw [smul_zero]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


theorem hammingNorm_smul [Zero α] [∀ i, SMulWithZero α (β i)] {k : α}
    (hk : ∀ i, IsSMulRegular (β i) k) (x : ∀ i, β i) : hammingNorm (k • x) = hammingNorm x :=
                                                             /-
                                                               α : Type u_1
                                                               ι : Type u_2
                                                               β : ι → Type u_3
                                                               inst✝⁴ : Fintype ι
                                                               inst✝³ : (i : ι) → DecidableEq (β i)
                                                               inst✝² : (i : ι) → Zero (β i)
                                                               inst✝¹ : Zero α
                                                               inst✝ : (i : ι) → SMulWithZero α (β i)
                                                               k : α
                                                               hk : ∀ (i : ι), IsSMulRegular (β i) k
                                                               x : (i : ι) → β i
                                                               i : ι
                                                               ⊢ Eq ((fun i c => HSMul.hSMul k c) i 0) 0
                                                             -/
  hammingNorm_comp (fun i (c : β i) => k • c) hk fun i => by simp_rw [smul_zero]
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- Corresponds to `dist_eq_norm`. -/
theorem hammingDist_eq_hammingNorm [∀ i, AddGroup (β i)] (x y : ∀ i, β i) :
    hammingDist x y = hammingNorm (x - y) := by
  /-
    ι : Type u_2
    β : ι → Type u_3
    inst✝² : Fintype ι
    inst✝¹ : (i : ι) → DecidableEq (β i)
    inst✝ : (i : ι) → AddGroup (β i)
    x y : (i : ι) → β i
    ⊢ Eq (hammingDist x y) (hammingNorm (HSub.hSub x y))
  -/
  simp_rw [hammingNorm, hammingDist, Pi.sub_apply, sub_ne_zero]
  /-
    🎉 no goals
  -/


/-- Type synonym for a Pi type which inherits the usual algebraic instances, but is equipped with
the Hamming metric and norm, instead of `Pi.normedAddCommGroup` which uses the sup norm. -/
def Hamming {ι : Type*} (β : ι → Type*) : Type _ :=
  ∀ i, β i


instance [∀ i, Inhabited (β i)] : Inhabited (Hamming β) :=
  ⟨fun _ => default⟩


instance [DecidableEq ι] [Fintype ι] [∀ i, Fintype (β i)] : Fintype (Hamming β) :=
  Pi.instFintype


instance [Inhabited ι] [∀ i, Nonempty (β i)] [Nontrivial (β default)] : Nontrivial (Hamming β) :=
  Pi.nontrivial


instance [Fintype ι] [∀ i, DecidableEq (β i)] : DecidableEq (Hamming β) :=
  Fintype.decidablePiFintype


instance [∀ i, Zero (β i)] : Zero (Hamming β) :=
  Pi.instZero


instance [∀ i, Neg (β i)] : Neg (Hamming β) :=
  Pi.instNeg


instance [∀ i, Add (β i)] : Add (Hamming β) :=
  Pi.instAdd


instance [∀ i, Sub (β i)] : Sub (Hamming β) :=
  Pi.instSub


instance [∀ i, SMul α (β i)] : SMul α (Hamming β) :=
  Pi.instSMul


instance [Zero α] [∀ i, Zero (β i)] [∀ i, SMulWithZero α (β i)] : SMulWithZero α (Hamming β) :=
  Pi.smulWithZero _


instance [∀ i, AddMonoid (β i)] : AddMonoid (Hamming β) :=
  Pi.addMonoid


instance [∀ i, AddCommMonoid (β i)] : AddCommMonoid (Hamming β) :=
  Pi.addCommMonoid


instance [∀ i, AddCommGroup (β i)] : AddCommGroup (Hamming β) :=
  Pi.addCommGroup


instance (α) [Semiring α] (β : ι → Type*) [∀ i, AddCommMonoid (β i)] [∀ i, Module α (β i)] :
    Module α (Hamming β) :=
  Pi.module _ _ _


/-- `Hamming.toHamming` is the identity function to the `Hamming` of a type. -/
@[match_pattern]
def toHamming : (∀ i, β i) ≃ Hamming β :=
  Equiv.refl _


/-- `Hamming.ofHamming` is the identity function from the `Hamming` of a type. -/
@[match_pattern]
def ofHamming : Hamming β ≃ ∀ i, β i :=
  Equiv.refl _


@[simp]
theorem toHamming_symm_eq : (@toHamming _ β).symm = ofHamming :=
  rfl


@[simp]
theorem ofHamming_symm_eq : (@ofHamming _ β).symm = toHamming :=
  rfl


@[simp]
theorem toHamming_ofHamming (x : Hamming β) : toHamming (ofHamming x) = x :=
  rfl


@[simp]
theorem ofHamming_toHamming (x : ∀ i, β i) : ofHamming (toHamming x) = x :=
  rfl


theorem toHamming_inj {x y : ∀ i, β i} : toHamming x = toHamming y ↔ x = y :=
  Iff.rfl


theorem ofHamming_inj {x y : Hamming β} : ofHamming x = ofHamming y ↔ x = y :=
  Iff.rfl


@[simp]
theorem toHamming_zero [∀ i, Zero (β i)] : toHamming (0 : ∀ i, β i) = 0 :=
  rfl


@[simp]
theorem ofHamming_zero [∀ i, Zero (β i)] : ofHamming (0 : Hamming β) = 0 :=
  rfl


@[simp]
theorem toHamming_neg [∀ i, Neg (β i)] {x : ∀ i, β i} : toHamming (-x) = -toHamming x :=
  rfl


@[simp]
theorem ofHamming_neg [∀ i, Neg (β i)] {x : Hamming β} : ofHamming (-x) = -ofHamming x :=
  rfl


@[simp]
theorem toHamming_add [∀ i, Add (β i)] {x y : ∀ i, β i} :
    toHamming (x + y) = toHamming x + toHamming y :=
  rfl


@[simp]
theorem ofHamming_add [∀ i, Add (β i)] {x y : Hamming β} :
    ofHamming (x + y) = ofHamming x + ofHamming y :=
  rfl


@[simp]
theorem toHamming_sub [∀ i, Sub (β i)] {x y : ∀ i, β i} :
    toHamming (x - y) = toHamming x - toHamming y :=
  rfl


@[simp]
theorem ofHamming_sub [∀ i, Sub (β i)] {x y : Hamming β} :
    ofHamming (x - y) = ofHamming x - ofHamming y :=
  rfl


@[simp]
theorem toHamming_smul [∀ i, SMul α (β i)] {r : α} {x : ∀ i, β i} :
    toHamming (r • x) = r • toHamming x :=
  rfl


@[simp]
theorem ofHamming_smul [∀ i, SMul α (β i)] {r : α} {x : Hamming β} :
    ofHamming (r • x) = r • ofHamming x :=
  rfl


instance : Dist (Hamming β) :=
  ⟨fun x y => hammingDist (ofHamming x) (ofHamming y)⟩


@[simp, push_cast]
theorem dist_eq_hammingDist (x y : Hamming β) :
    dist x y = hammingDist (ofHamming x) (ofHamming y) :=
  rfl


instance : PseudoMetricSpace (Hamming β) where
  dist_self := by
    /-
      α : Type u_1
      ι : Type u_2
      β : ι → Type u_3
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → DecidableEq (β i)
      ⊢ ∀ (x : Hamming β), Eq (Dist.dist x x) 0
    -/
    push_cast
    /-
      α : Type u_1
      ι : Type u_2
      β : ι → Type u_3
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → DecidableEq (β i)
      ⊢ ∀ (x : Hamming β), Eq (↑(hammingDist (Hamming.ofHamming x) (Hamming.ofHammin …
    -/
    exact mod_cast hammingDist_self
    /-
      🎉 no goals
    -/
  dist_comm := by
    /-
      α : Type u_1
      ι : Type u_2
      β : ι → Type u_3
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → DecidableEq (β i)
      ⊢ ∀ (x y : Hamming β), Eq (Dist.dist x y) (Dist.dist y x)
    -/
    push_cast
    /-
      α : Type u_1
      ι : Type u_2
      β : ι → Type u_3
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → DecidableEq (β i)
      ⊢ ∀ (x y : Hamming β), Eq ↑(hammingDist (Hamming.ofHamming x) (Hamming.ofHammi …
    -/
    exact mod_cast hammingDist_comm
    /-
      🎉 no goals
    -/
  dist_triangle := by
    /-
      α : Type u_1
      ι : Type u_2
      β : ι → Type u_3
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → DecidableEq (β i)
      ⊢ ∀ (x y z : Hamming β), LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x y) (Dis …
    -/
    push_cast
    /-
      α : Type u_1
      ι : Type u_2
      β : ι → Type u_3
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → DecidableEq (β i)
      ⊢ ∀ (x y z : Hamming β), LE.le (↑(hammingDist (Hamming.ofHamming x) (Hamming.o …
    -/
    exact mod_cast hammingDist_triangle
    /-
      🎉 no goals
    -/
  toUniformSpace := ⊥
  uniformity_dist := uniformity_dist_of_mem_uniformity _ _ fun s => by
    /-
      α : Type u_1
      ι : Type u_2
      β : ι → Type u_3
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → DecidableEq (β i)
      s : Set (Prod (Hamming β) (Hamming β))
      ⊢ Iff (Membership.mem (uniformity (Hamming β)) s) (Exists fun ε => And (GT.gt  …
    -/
    push_cast
    /-
      α : Type u_1
      ι : Type u_2
      β : ι → Type u_3
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → DecidableEq (β i)
      s : Set (Prod (Hamming β) (Hamming β))
      ⊢ Iff (Membership.mem (uniformity (Hamming β)) s) (Exists fun ε => And (GT.gt  …
    -/
    constructor
      /-
        case mp
        α : Type u_1
        ι : Type u_2
        β : ι → Type u_3
        inst✝¹ : Fintype ι
        inst✝ : (i : ι) → DecidableEq (β i)
        s : Set (Prod (Hamming β) (Hamming β))
        ⊢ Membership.mem (uniformity (Hamming β)) s → Exists fun ε => And (GT.gt ε 0)  …
      -/
    · refine fun hs => ⟨1, zero_lt_one, fun hab => ?_⟩
      /-
        case mp
        α : Type u_1
        ι : Type u_2
        β : ι → Type u_3
        inst✝¹ : Fintype ι
        inst✝ : (i : ι) → DecidableEq (β i)
        s : Set (Prod (Hamming β) (Hamming β))
        hs : Membership.mem (uniformity (Hamming β)) s
        a✝ b✝ : Hamming β
        hab : LT.lt (↑(hammingDist (Hamming.ofHamming a✝) (Hamming.ofHamming b✝))) 1
        ⊢ Membership.mem s { fst := a✝, snd := b✝ }
      -/
      rw_mod_cast [hammingDist_lt_one] at hab
      /-
        case mp
        α : Type u_1
        ι : Type u_2
        β : ι → Type u_3
        inst✝¹ : Fintype ι
        inst✝ : (i : ι) → DecidableEq (β i)
        s : Set (Prod (Hamming β) (Hamming β))
        hs : Membership.mem (uniformity (Hamming β)) s
        a✝ b✝ : Hamming β
        hab : Eq (Hamming.ofHamming a✝) (Hamming.ofHamming b✝)
        ⊢ Membership.mem s { fst := a✝, snd := b✝ }
      -/
      rw [ofHamming_inj, ← mem_idRel] at hab
      /-
        case mp
        α : Type u_1
        ι : Type u_2
        β : ι → Type u_3
        inst✝¹ : Fintype ι
        inst✝ : (i : ι) → DecidableEq (β i)
        s : Set (Prod (Hamming β) (Hamming β))
        hs : Membership.mem (uniformity (Hamming β)) s
        a✝ b✝ : Hamming β
        hab : Membership.mem idRel { fst := a✝, snd := b✝ }
        ⊢ Membership.mem s { fst := a✝, snd := b✝ }
      -/
      exact hs hab
      /-
        🎉 no goals
      -/
      /-
        case mpr
        α : Type u_1
        ι : Type u_2
        β : ι → Type u_3
        inst✝¹ : Fintype ι
        inst✝ : (i : ι) → DecidableEq (β i)
        s : Set (Prod (Hamming β) (Hamming β))
        ⊢ (Exists fun ε => And (GT.gt ε 0) (∀ {a b : Hamming β}, LT.lt (↑(hammingDist  …
      -/
    · rintro ⟨_, hε, hs⟩ ⟨_, _⟩ hab
      /-
        case mpr.intro.intro.mk
        α : Type u_1
        ι : Type u_2
        β : ι → Type u_3
        inst✝¹ : Fintype ι
        inst✝ : (i : ι) → DecidableEq (β i)
        s : Set (Prod (Hamming β) (Hamming β))
        w✝ : Real
        hε : GT.gt w✝ 0
        hs : ∀ {a b : Hamming β}, LT.lt (↑(hammingDist (Hamming.ofHamming a) (Hamming. …
        fst✝ snd✝ : Hamming β
        hab : Membership.mem idRel { fst := fst✝, snd := snd✝ }
        ⊢ Membership.mem s { fst := fst✝, snd := snd✝ }
      -/
      rw [mem_idRel] at hab
      /-
        case mpr.intro.intro.mk
        α : Type u_1
        ι : Type u_2
        β : ι → Type u_3
        inst✝¹ : Fintype ι
        inst✝ : (i : ι) → DecidableEq (β i)
        s : Set (Prod (Hamming β) (Hamming β))
        w✝ : Real
        hε : GT.gt w✝ 0
        hs : ∀ {a b : Hamming β}, LT.lt (↑(hammingDist (Hamming.ofHamming a) (Hamming. …
        fst✝ snd✝ : Hamming β
        hab : Eq fst✝ snd✝
        ⊢ Membership.mem s { fst := fst✝, snd := snd✝ }
      -/
      rw [hab]
      /-
        case mpr.intro.intro.mk
        α : Type u_1
        ι : Type u_2
        β : ι → Type u_3
        inst✝¹ : Fintype ι
        inst✝ : (i : ι) → DecidableEq (β i)
        s : Set (Prod (Hamming β) (Hamming β))
        w✝ : Real
        hε : GT.gt w✝ 0
        hs : ∀ {a b : Hamming β}, LT.lt (↑(hammingDist (Hamming.ofHamming a) (Hamming. …
        fst✝ snd✝ : Hamming β
        hab : Eq fst✝ snd✝
        ⊢ Membership.mem s { fst := snd✝, snd := snd✝ }
      -/
      refine hs (lt_of_eq_of_lt ?_ hε)
      /-
        case mpr.intro.intro.mk
        α : Type u_1
        ι : Type u_2
        β : ι → Type u_3
        inst✝¹ : Fintype ι
        inst✝ : (i : ι) → DecidableEq (β i)
        s : Set (Prod (Hamming β) (Hamming β))
        w✝ : Real
        hε : GT.gt w✝ 0
        hs : ∀ {a b : Hamming β}, LT.lt (↑(hammingDist (Hamming.ofHamming a) (Hamming. …
        fst✝ snd✝ : Hamming β
        hab : Eq fst✝ snd✝
        ⊢ Eq (↑(hammingDist (Hamming.ofHamming snd✝) (Hamming.ofHamming snd✝))) 0
      -/
      exact mod_cast hammingDist_self _
      /-
        🎉 no goals
      -/
  toBornology := ⟨⊥, bot_le⟩
  cobounded_sets := by
    /-
      α : Type u_1
      ι : Type u_2
      β : ι → Type u_3
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → DecidableEq (β i)
      ⊢ Eq (Bornology.cobounded (Hamming β)).sets (setOf fun s => Exists fun C => ∀  …
    -/
    ext
    /-
      case h
      α : Type u_1
      ι : Type u_2
      β : ι → Type u_3
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → DecidableEq (β i)
      x✝ : Set (Hamming β)
      ⊢ Iff (Membership.mem (Bornology.cobounded (Hamming β)).sets x✝) (Membership.m …
    -/
    push_cast
    /-
      case h
      α : Type u_1
      ι : Type u_2
      β : ι → Type u_3
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → DecidableEq (β i)
      x✝ : Set (Hamming β)
      ⊢ Iff (Membership.mem (Bornology.cobounded (Hamming β)).sets x✝) (Membership.m …
    -/
    refine iff_of_true (Filter.mem_sets.mpr Filter.mem_bot) ⟨Fintype.card ι, fun _ _ _ _ => ?_⟩
    /-
      case h
      α : Type u_1
      ι : Type u_2
      β : ι → Type u_3
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → DecidableEq (β i)
      x✝⁴ : Set (Hamming β)
      x✝³ : Hamming β
      x✝² : Membership.mem (HasCompl.compl x✝⁴) x✝³
      x✝¹ : Hamming β
      x✝ : Membership.mem (HasCompl.compl x✝⁴) x✝¹
      ⊢ LE.le ↑(hammingDist (Hamming.ofHamming x✝³) (Hamming.ofHamming x✝¹)) ↑(Finty …
    -/
    exact mod_cast hammingDist_le_card_fintype
    /-
      🎉 no goals
    -/


@[simp, push_cast]
theorem nndist_eq_hammingDist (x y : Hamming β) :
    nndist x y = hammingDist (ofHamming x) (ofHamming y) :=
  rfl

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): new instance

instance : DiscreteTopology (Hamming β) := ⟨rfl⟩


instance : MetricSpace (Hamming β) := .ofT0PseudoMetricSpace _


instance [∀ i, Zero (β i)] : Norm (Hamming β) :=
  ⟨fun x => hammingNorm (ofHamming x)⟩


@[simp, push_cast]
theorem norm_eq_hammingNorm [∀ i, Zero (β i)] (x : Hamming β) : ‖x‖ = hammingNorm (ofHamming x) :=
  rfl

-- Porting note: merged `SeminormedAddCommGroup` and `NormedAddCommGroup` instances


instance [∀ i, AddCommGroup (β i)] : NormedAddCommGroup (Hamming β) where
                /-
                  α : Type u_1
                  ι : Type u_2
                  β : ι → Type u_3
                  inst✝² : Fintype ι
                  inst✝¹ : (i : ι) → DecidableEq (β i)
                  inst✝ : (i : ι) → AddCommGroup (β i)
                  ⊢ ∀ (x y : Hamming β), Eq (Dist.dist x y) (Norm.norm (HSub.hSub x y))
                -/
  dist_eq := by push_cast; exact mod_cast hammingDist_eq_hammingNorm
                           /-
                             🎉 no goals
                           -/


@[simp, push_cast]
theorem nnnorm_eq_hammingNorm [∀ i, AddCommGroup (β i)] (x : Hamming β) :
    ‖x‖₊ = hammingNorm (ofHamming x) :=
  rfl


