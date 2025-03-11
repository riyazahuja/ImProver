/-- Segments in a vector space. -/
def segment (x y : E) : Set E :=
  { z : E | ∃ a b : 𝕜, 0 ≤ a ∧ 0 ≤ b ∧ a + b = 1 ∧ a • x + b • y = z }


/-- Open segment in a vector space. Note that `openSegment 𝕜 x x = {x}` instead of being `∅` when
the base semiring has some element between `0` and `1`. -/
def openSegment (x y : E) : Set E :=
  { z : E | ∃ a b : 𝕜, 0 < a ∧ 0 < b ∧ a + b = 1 ∧ a • x + b • y = z }


@[inherit_doc] scoped[Convex] notation (priority := high) "[" x "-[" 𝕜 "]" y "]" => segment 𝕜 x y


theorem segment_eq_image₂ (x y : E) :
    [x -[𝕜] y] =
      (fun p : 𝕜 × 𝕜 => p.1 • x + p.2 • y) '' { p | 0 ≤ p.1 ∧ 0 ≤ p.2 ∧ p.1 + p.2 = 1 } := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    x y : E
    ⊢ Eq (segment 𝕜 x y) (Set.image (fun p => HAdd.hAdd (HSMul.hSMul p.1 x) (HSMul …
  -/
  simp only [segment, image, Prod.exists, mem_setOf_eq, exists_prop, and_assoc]
  /-
    🎉 no goals
  -/


theorem openSegment_eq_image₂ (x y : E) :
    openSegment 𝕜 x y =
      (fun p : 𝕜 × 𝕜 => p.1 • x + p.2 • y) '' { p | 0 < p.1 ∧ 0 < p.2 ∧ p.1 + p.2 = 1 } := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    x y : E
    ⊢ Eq (openSegment 𝕜 x y) (Set.image (fun p => HAdd.hAdd (HSMul.hSMul p.1 x) (H …
  -/
  simp only [openSegment, image, Prod.exists, mem_setOf_eq, exists_prop, and_assoc]
  /-
    🎉 no goals
  -/


theorem segment_symm (x y : E) : [x -[𝕜] y] = [y -[𝕜] x] :=
  Set.ext fun _ =>
    ⟨fun ⟨a, b, ha, hb, hab, H⟩ => ⟨b, a, hb, ha, (add_comm _ _).trans hab, (add_comm _ _).trans H⟩,
      fun ⟨a, b, ha, hb, hab, H⟩ =>
      ⟨b, a, hb, ha, (add_comm _ _).trans hab, (add_comm _ _).trans H⟩⟩


theorem openSegment_symm (x y : E) : openSegment 𝕜 x y = openSegment 𝕜 y x :=
  Set.ext fun _ =>
    ⟨fun ⟨a, b, ha, hb, hab, H⟩ => ⟨b, a, hb, ha, (add_comm _ _).trans hab, (add_comm _ _).trans H⟩,
      fun ⟨a, b, ha, hb, hab, H⟩ =>
      ⟨b, a, hb, ha, (add_comm _ _).trans hab, (add_comm _ _).trans H⟩⟩


theorem openSegment_subset_segment (x y : E) : openSegment 𝕜 x y ⊆ [x -[𝕜] y] :=
  fun _ ⟨a, b, ha, hb, hab, hz⟩ => ⟨a, b, ha.le, hb.le, hab, hz⟩


theorem segment_subset_iff :
    [x -[𝕜] y] ⊆ s ↔ ∀ a b : 𝕜, 0 ≤ a → 0 ≤ b → a + b = 1 → a • x + b • y ∈ s :=
  ⟨fun H a b ha hb hab => H ⟨a, b, ha, hb, hab, rfl⟩, fun H _ ⟨a, b, ha, hb, hab, hz⟩ =>
    hz ▸ H a b ha hb hab⟩


theorem openSegment_subset_iff :
    openSegment 𝕜 x y ⊆ s ↔ ∀ a b : 𝕜, 0 < a → 0 < b → a + b = 1 → a • x + b • y ∈ s :=
  ⟨fun H a b ha hb hab => H ⟨a, b, ha, hb, hab, rfl⟩, fun H _ ⟨a, b, ha, hb, hab, hz⟩ =>
    hz ▸ H a b ha hb hab⟩


theorem left_mem_segment (x y : E) : x ∈ [x -[𝕜] y] :=
                                                /-
                                                  𝕜 : Type u_1
                                                  E : Type u_2
                                                  inst✝² : OrderedSemiring 𝕜
                                                  inst✝¹ : AddCommMonoid E
                                                  inst✝ : MulActionWithZero 𝕜 E
                                                  x y : E
                                                  ⊢ Eq (HAdd.hAdd (HSMul.hSMul 1 x) (HSMul.hSMul 0 y)) x
                                                -/
  ⟨1, 0, zero_le_one, le_refl 0, add_zero 1, by rw [zero_smul, one_smul, add_zero]⟩
                                                /-
                                                  🎉 no goals
                                                -/


theorem right_mem_segment (x y : E) : y ∈ [x -[𝕜] y] :=
  segment_symm 𝕜 y x ▸ left_mem_segment 𝕜 y x


@[simp]
theorem segment_same (x : E) : [x -[𝕜] x] = {x} :=
  Set.ext fun z =>
    ⟨fun ⟨a, b, _, _, hab, hz⟩ => by
      /-
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : OrderedSemiring 𝕜
        inst✝¹ : AddCommMonoid E
        inst✝ : Module 𝕜 E
        x z : E
        x✝ : Membership.mem (segment 𝕜 x x) z
        a b : 𝕜
        left✝¹ : LE.le 0 a
        left✝ : LE.le 0 b
        hab : Eq (HAdd.hAdd a b) 1
        hz : Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b x)) z
        ⊢ Membership.mem (Singleton.singleton x) z
      -/
      simpa only [(add_smul _ _ _).symm, mem_singleton_iff, hab, one_smul, eq_comm] using hz,
      /-
        🎉 no goals
      -/
      fun h => mem_singleton_iff.1 h ▸ left_mem_segment 𝕜 z z⟩


theorem insert_endpoints_openSegment (x y : E) :
    insert x (insert y (openSegment 𝕜 x y)) = [x -[𝕜] y] := by
  simp only [subset_antisymm_iff, insert_subset_iff, left_mem_segment, right_mem_segment,
    openSegment_subset_segment, true_and]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x y : E
    ⊢ HasSubset.Subset (segment 𝕜 x y) (Insert.insert x (Insert.insert y (openSegm …
  -/
  rintro z ⟨a, b, ha, hb, hab, rfl⟩
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x y : E
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (Insert.insert x (Insert.insert y (openSegment 𝕜 x y))) (HAdd …
  -/
  refine hb.eq_or_gt.imp ?_ fun hb' => ha.eq_or_gt.imp ?_ fun ha' => ?_
    /-
      case intro.intro.intro.intro.intro.refine_1
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      x y : E
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ Eq b 0 → Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y)) x
    -/
  · rintro rfl
    /-
      case intro.intro.intro.intro.intro.refine_1
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      x y : E
      a : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 0
      hab : Eq (HAdd.hAdd a 0) 1
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul 0 y)) x
    -/
    rw [← add_zero a, hab, one_smul, zero_smul, add_zero]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.refine_2
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      x y : E
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      hb' : LT.lt 0 b
      ⊢ Eq a 0 → Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y)) y
    -/
  · rintro rfl
    /-
      case intro.intro.intro.intro.intro.refine_2
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      x y : E
      b : 𝕜
      hb : LE.le 0 b
      hb' : LT.lt 0 b
      ha : LE.le 0 0
      hab : Eq (HAdd.hAdd 0 b) 1
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul 0 x) (HSMul.hSMul b y)) y
    -/
    rw [← zero_add b, hab, one_smul, zero_smul, zero_add]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.refine_3
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      x y : E
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      hb' : LT.lt 0 b
      ha' : LT.lt 0 a
      ⊢ Membership.mem (openSegment 𝕜 x y) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul …
    -/
  · exact ⟨a, b, ha', hb', hab, rfl⟩
    /-
      🎉 no goals
    -/


theorem mem_openSegment_of_ne_left_right (hx : x ≠ z) (hy : y ≠ z) (hz : z ∈ [x -[𝕜] y]) :
    z ∈ openSegment 𝕜 x y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x y z : E
    hx : Ne x z
    hy : Ne y z
    hz : Membership.mem (segment 𝕜 x y) z
    ⊢ Membership.mem (openSegment 𝕜 x y) z
  -/
  rw [← insert_endpoints_openSegment] at hz
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x y z : E
    hx : Ne x z
    hy : Ne y z
    hz : Membership.mem (Insert.insert x (Insert.insert y (openSegment 𝕜 x y))) z
    ⊢ Membership.mem (openSegment 𝕜 x y) z
  -/
  exact (hz.resolve_left hx.symm).resolve_left hy.symm
  /-
    🎉 no goals
  -/


theorem openSegment_subset_iff_segment_subset (hx : x ∈ s) (hy : y ∈ s) :
    openSegment 𝕜 x y ⊆ s ↔ [x -[𝕜] y] ⊆ s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s : Set E
    x y : E
    hx : Membership.mem s x
    hy : Membership.mem s y
    ⊢ Iff (HasSubset.Subset (openSegment 𝕜 x y) s) (HasSubset.Subset (segment 𝕜 x  …
  -/
  simp only [← insert_endpoints_openSegment, insert_subset_iff, *, true_and]
  /-
    🎉 no goals
  -/


@[simp]
theorem openSegment_same (x : E) : openSegment 𝕜 x x = {x} :=
  Set.ext fun z =>
    ⟨fun ⟨a, b, _, _, hab, hz⟩ => by
      /-
        𝕜 : Type u_1
        E : Type u_2
        inst✝⁴ : OrderedRing 𝕜
        inst✝³ : AddCommGroup E
        inst✝² : Module 𝕜 E
        inst✝¹ : Nontrivial 𝕜
        inst✝ : DenselyOrdered 𝕜
        x z : E
        x✝ : Membership.mem (openSegment 𝕜 x x) z
        a b : 𝕜
        left✝¹ : LT.lt 0 a
        left✝ : LT.lt 0 b
        hab : Eq (HAdd.hAdd a b) 1
        hz : Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b x)) z
        ⊢ Membership.mem (Singleton.singleton x) z
      -/
      simpa only [← add_smul, mem_singleton_iff, hab, one_smul, eq_comm] using hz,
      /-
        🎉 no goals
      -/
    fun h : z = x => by
      /-
        𝕜 : Type u_1
        E : Type u_2
        inst✝⁴ : OrderedRing 𝕜
        inst✝³ : AddCommGroup E
        inst✝² : Module 𝕜 E
        inst✝¹ : Nontrivial 𝕜
        inst✝ : DenselyOrdered 𝕜
        x z : E
        h : Eq z x
        ⊢ Membership.mem (openSegment 𝕜 x x) z
      -/
      obtain ⟨a, ha₀, ha₁⟩ := DenselyOrdered.dense (0 : 𝕜) 1 zero_lt_one
      /-
        case intro.intro
        𝕜 : Type u_1
        E : Type u_2
        inst✝⁴ : OrderedRing 𝕜
        inst✝³ : AddCommGroup E
        inst✝² : Module 𝕜 E
        inst✝¹ : Nontrivial 𝕜
        inst✝ : DenselyOrdered 𝕜
        x z : E
        h : Eq z x
        a : 𝕜
        ha₀ : LT.lt 0 a
        ha₁ : LT.lt a 1
        ⊢ Membership.mem (openSegment 𝕜 x x) z
      -/
      refine ⟨a, 1 - a, ha₀, sub_pos_of_lt ha₁, add_sub_cancel _ _, ?_⟩
      /-
        case intro.intro
        𝕜 : Type u_1
        E : Type u_2
        inst✝⁴ : OrderedRing 𝕜
        inst✝³ : AddCommGroup E
        inst✝² : Module 𝕜 E
        inst✝¹ : Nontrivial 𝕜
        inst✝ : DenselyOrdered 𝕜
        x z : E
        h : Eq z x
        a : 𝕜
        ha₀ : LT.lt 0 a
        ha₁ : LT.lt a 1
        ⊢ Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul (HSub.hSub 1 a) x)) z
      -/
      rw [← add_smul, add_sub_cancel, one_smul, h]⟩
      /-
        🎉 no goals
      -/


theorem segment_eq_image (x y : E) :
    [x -[𝕜] y] = (fun θ : 𝕜 => (1 - θ) • x + θ • y) '' Icc (0 : 𝕜) 1 :=
  Set.ext fun _ =>
    ⟨fun ⟨a, b, ha, hb, hab, hz⟩ =>
                                                              /-
                                                                𝕜 : Type u_1
                                                                E : Type u_2
                                                                inst✝² : OrderedRing 𝕜
                                                                inst✝¹ : AddCommGroup E
                                                                inst✝ : Module 𝕜 E
                                                                x y x✝¹ : E
                                                                x✝ : Membership.mem (segment 𝕜 x y) x✝¹
                                                                a b : 𝕜
                                                                ha : LE.le 0 a
                                                                hb : LE.le 0 b
                                                                hab : Eq (HAdd.hAdd a b) 1
                                                                hz : Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y)) x✝¹
                                                                ⊢ Eq ((fun θ => HAdd.hAdd (HSMul.hSMul (HSub.hSub (HAdd.hAdd a b) θ) x) (HSMul …
                                                              -/
      ⟨b, ⟨hb, hab ▸ le_add_of_nonneg_left ha⟩, hab ▸ hz ▸ by simp only [add_sub_cancel_right]⟩,
                                                              /-
                                                                🎉 no goals
                                                              -/
      fun ⟨θ, ⟨hθ₀, hθ₁⟩, hz⟩ => ⟨1 - θ, θ, sub_nonneg.2 hθ₁, hθ₀, sub_add_cancel _ _, hz⟩⟩


theorem openSegment_eq_image (x y : E) :
    openSegment 𝕜 x y = (fun θ : 𝕜 => (1 - θ) • x + θ • y) '' Ioo (0 : 𝕜) 1 :=
  Set.ext fun _ =>
    ⟨fun ⟨a, b, ha, hb, hab, hz⟩ =>
                                                             /-
                                                               𝕜 : Type u_1
                                                               E : Type u_2
                                                               inst✝² : OrderedRing 𝕜
                                                               inst✝¹ : AddCommGroup E
                                                               inst✝ : Module 𝕜 E
                                                               x y x✝¹ : E
                                                               x✝ : Membership.mem (openSegment 𝕜 x y) x✝¹
                                                               a b : 𝕜
                                                               ha : LT.lt 0 a
                                                               hb : LT.lt 0 b
                                                               hab : Eq (HAdd.hAdd a b) 1
                                                               hz : Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y)) x✝¹
                                                               ⊢ Eq ((fun θ => HAdd.hAdd (HSMul.hSMul (HSub.hSub (HAdd.hAdd a b) θ) x) (HSMul …
                                                             -/
      ⟨b, ⟨hb, hab ▸ lt_add_of_pos_left _ ha⟩, hab ▸ hz ▸ by simp only [add_sub_cancel_right]⟩,
                                                             /-
                                                               🎉 no goals
                                                             -/
      fun ⟨θ, ⟨hθ₀, hθ₁⟩, hz⟩ => ⟨1 - θ, θ, sub_pos.2 hθ₁, hθ₀, sub_add_cancel _ _, hz⟩⟩


theorem segment_eq_image' (x y : E) :
    [x -[𝕜] y] = (fun θ : 𝕜 => x + θ • (y - x)) '' Icc (0 : 𝕜) 1 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y : E
    ⊢ Eq (segment 𝕜 x y) (Set.image (fun θ => HAdd.hAdd x (HSMul.hSMul θ (HSub.hSu …
  -/
  convert segment_eq_image 𝕜 x y using 2
  /-
    case h.e'_3.h
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y : E
    a✝¹ : 𝕜
    a✝ : Membership.mem (Set.Icc 0 1) a✝¹
    ⊢ Eq (HAdd.hAdd x (HSMul.hSMul a✝¹ (HSub.hSub y x))) (HAdd.hAdd (HSMul.hSMul ( …
  -/
  simp only [smul_sub, sub_smul, one_smul]
  /-
    case h.e'_3.h
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y : E
    a✝¹ : 𝕜
    a✝ : Membership.mem (Set.Icc 0 1) a✝¹
    ⊢ Eq (HAdd.hAdd x (HSub.hSub (HSMul.hSMul a✝¹ y) (HSMul.hSMul a✝¹ x))) (HAdd.h …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


theorem openSegment_eq_image' (x y : E) :
    openSegment 𝕜 x y = (fun θ : 𝕜 => x + θ • (y - x)) '' Ioo (0 : 𝕜) 1 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y : E
    ⊢ Eq (openSegment 𝕜 x y) (Set.image (fun θ => HAdd.hAdd x (HSMul.hSMul θ (HSub …
  -/
  convert openSegment_eq_image 𝕜 x y using 2
  /-
    case h.e'_3.h
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y : E
    a✝¹ : 𝕜
    a✝ : Membership.mem (Set.Ioo 0 1) a✝¹
    ⊢ Eq (HAdd.hAdd x (HSMul.hSMul a✝¹ (HSub.hSub y x))) (HAdd.hAdd (HSMul.hSMul ( …
  -/
  simp only [smul_sub, sub_smul, one_smul]
  /-
    case h.e'_3.h
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y : E
    a✝¹ : 𝕜
    a✝ : Membership.mem (Set.Ioo 0 1) a✝¹
    ⊢ Eq (HAdd.hAdd x (HSub.hSub (HSMul.hSMul a✝¹ y) (HSMul.hSMul a✝¹ x))) (HAdd.h …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


theorem segment_eq_image_lineMap (x y : E) : [x -[𝕜] y] =
    AffineMap.lineMap x y '' Icc (0 : 𝕜) 1 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y : E
    ⊢ Eq (segment 𝕜 x y) (Set.image (⇑(AffineMap.lineMap x y)) (Set.Icc 0 1))
  -/
  convert segment_eq_image 𝕜 x y using 2
  /-
    case h.e'_3.h
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y : E
    a✝¹ : 𝕜
    a✝ : Membership.mem (Set.Icc 0 1) a✝¹
    ⊢ Eq ((AffineMap.lineMap x y) a✝¹) (HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 a✝¹) x …
  -/
  exact AffineMap.lineMap_apply_module _ _ _
  /-
    🎉 no goals
  -/


theorem openSegment_eq_image_lineMap (x y : E) :
    openSegment 𝕜 x y = AffineMap.lineMap x y '' Ioo (0 : 𝕜) 1 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y : E
    ⊢ Eq (openSegment 𝕜 x y) (Set.image (⇑(AffineMap.lineMap x y)) (Set.Ioo 0 1))
  -/
  convert openSegment_eq_image 𝕜 x y using 2
  /-
    case h.e'_3.h
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y : E
    a✝¹ : 𝕜
    a✝ : Membership.mem (Set.Ioo 0 1) a✝¹
    ⊢ Eq ((AffineMap.lineMap x y) a✝¹) (HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 a✝¹) x …
  -/
  exact AffineMap.lineMap_apply_module _ _ _
  /-
    🎉 no goals
  -/


@[simp]
theorem image_segment (f : E →ᵃ[𝕜] F) (a b : E) : f '' [a -[𝕜] b] = [f a -[𝕜] f b] :=
  Set.ext fun x => by
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : OrderedRing 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : AddCommGroup F
      inst✝¹ : Module 𝕜 E
      inst✝ : Module 𝕜 F
      f : AffineMap 𝕜 E F
      a b : E
      x : F
      ⊢ Iff (Membership.mem (Set.image (⇑f) (segment 𝕜 a b)) x) (Membership.mem (seg …
    -/
    simp_rw [segment_eq_image_lineMap, mem_image, exists_exists_and_eq_and, AffineMap.apply_lineMap]
    /-
      🎉 no goals
    -/


@[simp]
theorem image_openSegment (f : E →ᵃ[𝕜] F) (a b : E) :
    f '' openSegment 𝕜 a b = openSegment 𝕜 (f a) (f b) :=
  Set.ext fun x => by
    simp_rw [openSegment_eq_image_lineMap, mem_image, exists_exists_and_eq_and,
      AffineMap.apply_lineMap]


@[simp]
theorem vadd_segment [AddTorsor G E] [VAddCommClass G E E] (a : G) (b c : E) :
    a +ᵥ [b -[𝕜] c] = [a +ᵥ b -[𝕜] a +ᵥ c] :=
  image_segment 𝕜 ⟨_, LinearMap.id, fun _ _ => vadd_comm _ _ _⟩ b c


@[simp]
theorem vadd_openSegment [AddTorsor G E] [VAddCommClass G E E] (a : G) (b c : E) :
    a +ᵥ openSegment 𝕜 b c = openSegment 𝕜 (a +ᵥ b) (a +ᵥ c) :=
  image_openSegment 𝕜 ⟨_, LinearMap.id, fun _ _ => vadd_comm _ _ _⟩ b c


@[simp]
theorem mem_segment_translate (a : E) {x b c} : a + x ∈ [a + b -[𝕜] a + c] ↔ x ∈ [b -[𝕜] c] := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    a x b c : E
    ⊢ Iff (Membership.mem (segment 𝕜 (HAdd.hAdd a b) (HAdd.hAdd a c)) (HAdd.hAdd a …
  -/
  simp_rw [← vadd_eq_add, ← vadd_segment, vadd_mem_vadd_set_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_openSegment_translate (a : E) {x b c : E} :
    a + x ∈ openSegment 𝕜 (a + b) (a + c) ↔ x ∈ openSegment 𝕜 b c := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    a x b c : E
    ⊢ Iff (Membership.mem (openSegment 𝕜 (HAdd.hAdd a b) (HAdd.hAdd a c)) (HAdd.hA …
  -/
  simp_rw [← vadd_eq_add, ← vadd_openSegment, vadd_mem_vadd_set_iff]
  /-
    🎉 no goals
  -/


theorem segment_translate_preimage (a b c : E) :
    (fun x => a + x) ⁻¹' [a + b -[𝕜] a + c] = [b -[𝕜] c] :=
  Set.ext fun _ => mem_segment_translate 𝕜 a


theorem openSegment_translate_preimage (a b c : E) :
    (fun x => a + x) ⁻¹' openSegment 𝕜 (a + b) (a + c) = openSegment 𝕜 b c :=
  Set.ext fun _ => mem_openSegment_translate 𝕜 a


theorem segment_translate_image (a b c : E) : (fun x => a + x) '' [b -[𝕜] c] = [a + b -[𝕜] a + c] :=
  segment_translate_preimage 𝕜 a b c ▸ image_preimage_eq _ <| add_left_surjective a


theorem openSegment_translate_image (a b c : E) :
    (fun x => a + x) '' openSegment 𝕜 b c = openSegment 𝕜 (a + b) (a + c) :=
  openSegment_translate_preimage 𝕜 a b c ▸ image_preimage_eq _ <| add_left_surjective a


lemma segment_inter_eq_endpoint_of_linearIndependent_sub
    {c x y : E} (h : LinearIndependent 𝕜 ![x - c, y - c]) :
    [c -[𝕜] x] ∩ [c -[𝕜] y] = {c} := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    c x y : E
    h : LinearIndependent 𝕜 (Matrix.vecCons (HSub.hSub x c) (Matrix.vecCons (HSub. …
    ⊢ Eq (Inter.inter (segment 𝕜 c x) (segment 𝕜 c y)) (Singleton.singleton c)
  -/
  apply Subset.antisymm; swap
    /-
      case h₂
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedRing 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      c x y : E
      h : LinearIndependent 𝕜 (Matrix.vecCons (HSub.hSub x c) (Matrix.vecCons (HSub. …
      ⊢ HasSubset.Subset (Singleton.singleton c) (Inter.inter (segment 𝕜 c x) (segme …
    -/
  · simp [singleton_subset_iff, left_mem_segment]
    /-
      🎉 no goals
    -/
  /-
    case h₁
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    c x y : E
    h : LinearIndependent 𝕜 (Matrix.vecCons (HSub.hSub x c) (Matrix.vecCons (HSub. …
    ⊢ HasSubset.Subset (Inter.inter (segment 𝕜 c x) (segment 𝕜 c y)) (Singleton.si …
  -/
  intro z ⟨hzt, hzs⟩
  /-
    case h₁
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    c x y : E
    h : LinearIndependent 𝕜 (Matrix.vecCons (HSub.hSub x c) (Matrix.vecCons (HSub. …
    z : E
    hzt : Membership.mem (segment 𝕜 c x) z
    hzs : Membership.mem (segment 𝕜 c y) z
    ⊢ Membership.mem (Singleton.singleton c) z
  -/
  rw [segment_eq_image, mem_image] at hzt hzs
  /-
    case h₁
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    c x y : E
    h : LinearIndependent 𝕜 (Matrix.vecCons (HSub.hSub x c) (Matrix.vecCons (HSub. …
    z : E
    hzt : Exists fun x_1 => And (Membership.mem (Set.Icc 0 1) x_1) (Eq (HAdd.hAdd  …
    hzs : Exists fun x => And (Membership.mem (Set.Icc 0 1) x) (Eq (HAdd.hAdd (HSM …
    ⊢ Membership.mem (Singleton.singleton c) z
  -/
  rcases hzt with ⟨p, ⟨p0, p1⟩, rfl⟩
  /-
    case h₁.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    c x y : E
    h : LinearIndependent 𝕜 (Matrix.vecCons (HSub.hSub x c) (Matrix.vecCons (HSub. …
    p : 𝕜
    p0 : LE.le 0 p
    p1 : LE.le p 1
    hzs : Exists fun x_1 => And (Membership.mem (Set.Icc 0 1) x_1) (Eq (HAdd.hAdd  …
    ⊢ Membership.mem (Singleton.singleton c) (HAdd.hAdd (HSMul.hSMul (HSub.hSub 1  …
  -/
  rcases hzs with ⟨q, ⟨q0, q1⟩, H⟩
  /-
    case h₁.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    c x y : E
    h : LinearIndependent 𝕜 (Matrix.vecCons (HSub.hSub x c) (Matrix.vecCons (HSub. …
    p : 𝕜
    p0 : LE.le 0 p
    p1 : LE.le p 1
    q : 𝕜
    H : Eq (HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 q) c) (HSMul.hSMul q y)) (HAdd.hAd …
    q0 : LE.le 0 q
    q1 : LE.le q 1
    ⊢ Membership.mem (Singleton.singleton c) (HAdd.hAdd (HSMul.hSMul (HSub.hSub 1  …
  -/
  have Hx : x = (x - c) + c := by abel
  /-
    case h₁.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    c x y : E
    h : LinearIndependent 𝕜 (Matrix.vecCons (HSub.hSub x c) (Matrix.vecCons (HSub. …
    p : 𝕜
    p0 : LE.le 0 p
    p1 : LE.le p 1
    q : 𝕜
    H : Eq (HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 q) c) (HSMul.hSMul q y)) (HAdd.hAd …
    q0 : LE.le 0 q
    q1 : LE.le q 1
    Hx : Eq x (HAdd.hAdd (HSub.hSub x c) c)
    ⊢ Membership.mem (Singleton.singleton c) (HAdd.hAdd (HSMul.hSMul (HSub.hSub 1  …
  -/
  have Hy : y = (y - c) + c := by abel
  /-
    case h₁.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    c x y : E
    h : LinearIndependent 𝕜 (Matrix.vecCons (HSub.hSub x c) (Matrix.vecCons (HSub. …
    p : 𝕜
    p0 : LE.le 0 p
    p1 : LE.le p 1
    q : 𝕜
    H : Eq (HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 q) c) (HSMul.hSMul q y)) (HAdd.hAd …
    q0 : LE.le 0 q
    q1 : LE.le q 1
    Hx : Eq x (HAdd.hAdd (HSub.hSub x c) c)
    Hy : Eq y (HAdd.hAdd (HSub.hSub y c) c)
    ⊢ Membership.mem (Singleton.singleton c) (HAdd.hAdd (HSMul.hSMul (HSub.hSub 1  …
  -/
  rw [Hx, Hy, smul_add, smul_add] at H
  have : c + q • (y - c) = c + p • (x - c) := by
    convert H using 1 <;> simp [sub_smul]
  /-
    case h₁.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    c x y : E
    h : LinearIndependent 𝕜 (Matrix.vecCons (HSub.hSub x c) (Matrix.vecCons (HSub. …
    p : 𝕜
    p0 : LE.le 0 p
    p1 : LE.le p 1
    q : 𝕜
    H : Eq (HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 q) c) (HAdd.hAdd (HSMul.hSMul q (H …
    q0 : LE.le 0 q
    q1 : LE.le q 1
    Hx : Eq x (HAdd.hAdd (HSub.hSub x c) c)
    Hy : Eq y (HAdd.hAdd (HSub.hSub y c) c)
    this : Eq (HAdd.hAdd c (HSMul.hSMul q (HSub.hSub y c))) (HAdd.hAdd c (HSMul.hS …
    ⊢ Membership.mem (Singleton.singleton c) (HAdd.hAdd (HSMul.hSMul (HSub.hSub 1  …
  -/
  obtain ⟨rfl, rfl⟩ : p = 0 ∧ q = 0 := h.eq_zero_of_pair' ((add_right_inj c).1 this).symm
  /-
    case h₁.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    c x y : E
    h : LinearIndependent 𝕜 (Matrix.vecCons (HSub.hSub x c) (Matrix.vecCons (HSub. …
    Hx : Eq x (HAdd.hAdd (HSub.hSub x c) c)
    Hy : Eq y (HAdd.hAdd (HSub.hSub y c) c)
    p0 : LE.le 0 0
    p1 : LE.le 0 1
    q0 : LE.le 0 0
    q1 : LE.le 0 1
    H : Eq (HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 0) c) (HAdd.hAdd (HSMul.hSMul 0 (H …
    this : Eq (HAdd.hAdd c (HSMul.hSMul 0 (HSub.hSub y c))) (HAdd.hAdd c (HSMul.hS …
    ⊢ Membership.mem (Singleton.singleton c) (HAdd.hAdd (HSMul.hSMul (HSub.hSub 1  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem sameRay_of_mem_segment [StrictOrderedCommRing 𝕜] [AddCommGroup E] [Module 𝕜 E] {x y z : E}
    (h : x ∈ [y -[𝕜] z]) : SameRay 𝕜 (x - y) (z - x) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : StrictOrderedCommRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y z : E
    h : Membership.mem (segment 𝕜 y z) x
    ⊢ SameRay 𝕜 (HSub.hSub x y) (HSub.hSub z x)
  -/
  rw [segment_eq_image'] at h
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : StrictOrderedCommRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y z : E
    h : Membership.mem (Set.image (fun θ => HAdd.hAdd y (HSMul.hSMul θ (HSub.hSub  …
    ⊢ SameRay 𝕜 (HSub.hSub x y) (HSub.hSub z x)
  -/
  rcases h with ⟨θ, ⟨hθ₀, hθ₁⟩, rfl⟩
  simpa only [add_sub_cancel_left, ← sub_sub, sub_smul, one_smul] using
    (SameRay.sameRay_nonneg_smul_left (z - y) hθ₀).nonneg_smul_right (sub_nonneg.2 hθ₁)


lemma segment_inter_eq_endpoint_of_linearIndependent_of_ne [OrderedCommRing 𝕜] [NoZeroDivisors 𝕜]
    [AddCommGroup E] [Module 𝕜 E]
    {x y : E} (h : LinearIndependent 𝕜 ![x, y]) {s t : 𝕜} (hs : s ≠ t) (c : E) :
    [c + x -[𝕜] c + t • y] ∩ [c + x -[𝕜] c + s • y] = {c + x} := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : OrderedCommRing 𝕜
    inst✝² : NoZeroDivisors 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y : E
    h : LinearIndependent 𝕜 (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
    s t : 𝕜
    hs : Ne s t
    c : E
    ⊢ Eq (Inter.inter (segment 𝕜 (HAdd.hAdd c x) (HAdd.hAdd c (HSMul.hSMul t y)))  …
  -/
  apply segment_inter_eq_endpoint_of_linearIndependent_sub
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : OrderedCommRing 𝕜
    inst✝² : NoZeroDivisors 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y : E
    h : LinearIndependent 𝕜 (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
    s t : 𝕜
    hs : Ne s t
    c : E
    ⊢ LinearIndependent 𝕜 (Matrix.vecCons (HSub.hSub (HAdd.hAdd c (HSMul.hSMul t y …
  -/
  simp only [add_sub_add_left_eq_sub]
  suffices H : LinearIndependent 𝕜 ![(-1 : 𝕜) • x + t • y, (-1 : 𝕜) • x + s • y] by
    convert H using 1; simp only [neg_smul, one_smul]; abel_nf
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : OrderedCommRing 𝕜
    inst✝² : NoZeroDivisors 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y : E
    h : LinearIndependent 𝕜 (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
    s t : 𝕜
    hs : Ne s t
    c : E
    ⊢ LinearIndependent 𝕜 (Matrix.vecCons (HAdd.hAdd (HSMul.hSMul (-1) x) (HSMul.h …
  -/
  apply h.linear_combination_pair_of_det_ne_zero
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : OrderedCommRing 𝕜
    inst✝² : NoZeroDivisors 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y : E
    h : LinearIndependent 𝕜 (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
    s t : 𝕜
    hs : Ne s t
    c : E
    ⊢ Ne (HSub.hSub (HMul.hMul (-1) s) (HMul.hMul t (-1))) 0
  -/
  contrapose! hs
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : OrderedCommRing 𝕜
    inst✝² : NoZeroDivisors 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y : E
    h : LinearIndependent 𝕜 (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
    s t : 𝕜
    c : E
    hs : Eq (HSub.hSub (HMul.hMul (-1) s) (HMul.hMul t (-1))) 0
    ⊢ Eq s t
  -/
  apply Eq.symm
  simpa [neg_mul, one_mul, mul_neg, mul_one, sub_neg_eq_add, add_comm _ t,
    ← sub_eq_add_neg, sub_eq_zero] using hs


theorem midpoint_mem_segment [Invertible (2 : 𝕜)] (x y : E) : midpoint 𝕜 x y ∈ [x -[𝕜] y] := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : LinearOrderedRing 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Invertible 2
    x y : E
    ⊢ Membership.mem (segment 𝕜 x y) (midpoint 𝕜 x y)
  -/
  rw [segment_eq_image_lineMap]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : LinearOrderedRing 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Invertible 2
    x y : E
    ⊢ Membership.mem (Set.image (⇑(AffineMap.lineMap x y)) (Set.Icc 0 1)) (midpoin …
  -/
  exact ⟨⅟ 2, ⟨invOf_nonneg.mpr zero_le_two, invOf_le_one one_le_two⟩, rfl⟩
  /-
    🎉 no goals
  -/


theorem mem_segment_sub_add [Invertible (2 : 𝕜)] (x y : E) : x ∈ [x - y -[𝕜] x + y] := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : LinearOrderedRing 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Invertible 2
    x y : E
    ⊢ Membership.mem (segment 𝕜 (HSub.hSub x y) (HAdd.hAdd x y)) x
  -/
  convert @midpoint_mem_segment 𝕜 _ _ _ _ _ (x - y) (x + y)
  /-
    case h.e'_5
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : LinearOrderedRing 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Invertible 2
    x y : E
    ⊢ Eq x (midpoint 𝕜 (HSub.hSub x y) (HAdd.hAdd x y))
  -/
  rw [midpoint_sub_add]
  /-
    🎉 no goals
  -/


theorem mem_segment_add_sub [Invertible (2 : 𝕜)] (x y : E) : x ∈ [x + y -[𝕜] x - y] := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : LinearOrderedRing 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Invertible 2
    x y : E
    ⊢ Membership.mem (segment 𝕜 (HAdd.hAdd x y) (HSub.hSub x y)) x
  -/
  convert @midpoint_mem_segment 𝕜 _ _ _ _ _ (x + y) (x - y)
  /-
    case h.e'_5
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : LinearOrderedRing 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Invertible 2
    x y : E
    ⊢ Eq x (midpoint 𝕜 (HAdd.hAdd x y) (HSub.hSub x y))
  -/
  rw [midpoint_add_sub]
  /-
    🎉 no goals
  -/


@[simp]
theorem left_mem_openSegment_iff [DenselyOrdered 𝕜] [NoZeroSMulDivisors 𝕜 E] :
    x ∈ openSegment 𝕜 x y ↔ x = y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : LinearOrderedRing 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    x y : E
    inst✝¹ : DenselyOrdered 𝕜
    inst✝ : NoZeroSMulDivisors 𝕜 E
    ⊢ Iff (Membership.mem (openSegment 𝕜 x y) x) (Eq x y)
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : LinearOrderedRing 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      x y : E
      inst✝¹ : DenselyOrdered 𝕜
      inst✝ : NoZeroSMulDivisors 𝕜 E
      ⊢ Membership.mem (openSegment 𝕜 x y) x → Eq x y
    -/
  · rintro ⟨a, b, _, hb, hab, hx⟩
    /-
      case mp.intro.intro.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : LinearOrderedRing 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      x y : E
      inst✝¹ : DenselyOrdered 𝕜
      inst✝ : NoZeroSMulDivisors 𝕜 E
      a b : 𝕜
      left✝ : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      hx : Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y)) x
      ⊢ Eq x y
    -/
    refine smul_right_injective _ hb.ne' ((add_right_inj (a • x)).1 ?_)
    /-
      case mp.intro.intro.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : LinearOrderedRing 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      x y : E
      inst✝¹ : DenselyOrdered 𝕜
      inst✝ : NoZeroSMulDivisors 𝕜 E
      a b : 𝕜
      left✝ : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      hx : Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y)) x
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul a x) ((fun x => HSMul.hSMul b x) x)) (HAdd.hAdd ( …
    -/
    rw [hx, ← add_smul, hab, one_smul]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : LinearOrderedRing 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      x y : E
      inst✝¹ : DenselyOrdered 𝕜
      inst✝ : NoZeroSMulDivisors 𝕜 E
      ⊢ Eq x y → Membership.mem (openSegment 𝕜 x y) x
    -/
  · rintro rfl
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : LinearOrderedRing 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      x : E
      inst✝¹ : DenselyOrdered 𝕜
      inst✝ : NoZeroSMulDivisors 𝕜 E
      ⊢ Membership.mem (openSegment 𝕜 x x) x
    -/
    rw [openSegment_same]
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : LinearOrderedRing 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      x : E
      inst✝¹ : DenselyOrdered 𝕜
      inst✝ : NoZeroSMulDivisors 𝕜 E
      ⊢ Membership.mem (Singleton.singleton x) x
    -/
    exact mem_singleton _
    /-
      🎉 no goals
    -/


@[simp]
theorem right_mem_openSegment_iff [DenselyOrdered 𝕜] [NoZeroSMulDivisors 𝕜 E] :
                                        /-
                                          𝕜 : Type u_1
                                          E : Type u_2
                                          inst✝⁴ : LinearOrderedRing 𝕜
                                          inst✝³ : AddCommGroup E
                                          inst✝² : Module 𝕜 E
                                          x y : E
                                          inst✝¹ : DenselyOrdered 𝕜
                                          inst✝ : NoZeroSMulDivisors 𝕜 E
                                          ⊢ Iff (Membership.mem (openSegment 𝕜 x y) y) (Eq x y)
                                        -/
    y ∈ openSegment 𝕜 x y ↔ x = y := by rw [openSegment_symm, left_mem_openSegment_iff, eq_comm]
                                        /-
                                          🎉 no goals
                                        -/


theorem mem_segment_iff_div :
    x ∈ [y -[𝕜] z] ↔
      ∃ a b : 𝕜, 0 ≤ a ∧ 0 ≤ b ∧ 0 < a + b ∧ (a / (a + b)) • y + (b / (a + b)) • z = x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedSemifield 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y z : E
    ⊢ Iff (Membership.mem (segment 𝕜 y z) x) (Exists fun a => Exists fun b => And  …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedSemifield 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      x y z : E
      ⊢ Membership.mem (segment 𝕜 y z) x → Exists fun a => Exists fun b => And (LE.l …
    -/
  · rintro ⟨a, b, ha, hb, hab, rfl⟩
    /-
      case mp.intro.intro.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedSemifield 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      y z : E
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ Exists fun a_1 => Exists fun b_1 => And (LE.le 0 a_1) (And (LE.le 0 b_1) (An …
    -/
    use a, b, ha, hb
    /-
      case right
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedSemifield 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      y z : E
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ And (LT.lt 0 (HAdd.hAdd a b)) (Eq (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv a (HAdd …
    -/
    simp [*]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedSemifield 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      x y z : E
      ⊢ (Exists fun a => Exists fun b => And (LE.le 0 a) (And (LE.le 0 b) (And (LT.l …
    -/
  · rintro ⟨a, b, ha, hb, hab, rfl⟩
    /-
      case mpr.intro.intro.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedSemifield 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      y z : E
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : LT.lt 0 (HAdd.hAdd a b)
      ⊢ Membership.mem (segment 𝕜 y z) (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv a (HAdd.hA …
    -/
    refine ⟨a / (a + b), b / (a + b), by positivity, by positivity, ?_, rfl⟩
    /-
      case mpr.intro.intro.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedSemifield 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      y z : E
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : LT.lt 0 (HAdd.hAdd a b)
      ⊢ Eq (HAdd.hAdd (HDiv.hDiv a (HAdd.hAdd a b)) (HDiv.hDiv b (HAdd.hAdd a b))) 1
    -/
    rw [← add_div, div_self hab.ne']
    /-
      🎉 no goals
    -/


theorem mem_openSegment_iff_div : x ∈ openSegment 𝕜 y z ↔
    ∃ a b : 𝕜, 0 < a ∧ 0 < b ∧ (a / (a + b)) • y + (b / (a + b)) • z = x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedSemifield 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y z : E
    ⊢ Iff (Membership.mem (openSegment 𝕜 y z) x) (Exists fun a => Exists fun b =>  …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedSemifield 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      x y z : E
      ⊢ Membership.mem (openSegment 𝕜 y z) x → Exists fun a => Exists fun b => And ( …
    -/
  · rintro ⟨a, b, ha, hb, hab, rfl⟩
    /-
      case mp.intro.intro.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedSemifield 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      y z : E
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ Exists fun a_1 => Exists fun b_1 => And (LT.lt 0 a_1) (And (LT.lt 0 b_1) (Eq …
    -/
    use a, b, ha, hb
    /-
      case right
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedSemifield 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      y z : E
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv a (HAdd.hAdd a b)) y) (HSMul.hSMul (HD …
    -/
    rw [hab, div_one, div_one]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedSemifield 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      x y z : E
      ⊢ (Exists fun a => Exists fun b => And (LT.lt 0 a) (And (LT.lt 0 b) (Eq (HAdd. …
    -/
  · rintro ⟨a, b, ha, hb, rfl⟩
    /-
      case mpr.intro.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedSemifield 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      y z : E
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      ⊢ Membership.mem (openSegment 𝕜 y z) (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv a (HAd …
    -/
    have hab : 0 < a + b := add_pos' ha hb
    /-
      case mpr.intro.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedSemifield 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      y z : E
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : LT.lt 0 (HAdd.hAdd a b)
      ⊢ Membership.mem (openSegment 𝕜 y z) (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv a (HAd …
    -/
    refine ⟨a / (a + b), b / (a + b), by positivity, by positivity, ?_, rfl⟩
    /-
      case mpr.intro.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedSemifield 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      y z : E
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : LT.lt 0 (HAdd.hAdd a b)
      ⊢ Eq (HAdd.hAdd (HDiv.hDiv a (HAdd.hAdd a b)) (HDiv.hDiv b (HAdd.hAdd a b))) 1
    -/
    rw [← add_div, div_self hab.ne']
    /-
      🎉 no goals
    -/


theorem mem_segment_iff_sameRay : x ∈ [y -[𝕜] z] ↔ SameRay 𝕜 (x - y) (z - x) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y z : E
    ⊢ Iff (Membership.mem (segment 𝕜 y z) x) (SameRay 𝕜 (HSub.hSub x y) (HSub.hSub …
  -/
  refine ⟨sameRay_of_mem_segment, fun h => ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y z : E
    h : SameRay 𝕜 (HSub.hSub x y) (HSub.hSub z x)
    ⊢ Membership.mem (segment 𝕜 y z) x
  -/
  rcases h.exists_eq_smul_add with ⟨a, b, ha, hb, hab, hxy, hzx⟩
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y z : E
    h : SameRay 𝕜 (HSub.hSub x y) (HSub.hSub z x)
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    hxy : Eq (HSub.hSub x y) (HSMul.hSMul a (HAdd.hAdd (HSub.hSub x y) (HSub.hSub  …
    hzx : Eq (HSub.hSub z x) (HSMul.hSMul b (HAdd.hAdd (HSub.hSub x y) (HSub.hSub  …
    ⊢ Membership.mem (segment 𝕜 y z) x
  -/
  rw [add_comm, sub_add_sub_cancel] at hxy hzx
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y z : E
    h : SameRay 𝕜 (HSub.hSub x y) (HSub.hSub z x)
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    hxy : Eq (HSub.hSub x y) (HSMul.hSMul a (HSub.hSub z y))
    hzx : Eq (HSub.hSub z x) (HSMul.hSMul b (HSub.hSub z y))
    ⊢ Membership.mem (segment 𝕜 y z) x
  -/
  rw [← mem_segment_translate _ (-x), neg_add_cancel]
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y z : E
    h : SameRay 𝕜 (HSub.hSub x y) (HSub.hSub z x)
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    hxy : Eq (HSub.hSub x y) (HSMul.hSMul a (HSub.hSub z y))
    hzx : Eq (HSub.hSub z x) (HSMul.hSMul b (HSub.hSub z y))
    ⊢ Membership.mem (segment 𝕜 (HAdd.hAdd (Neg.neg x) y) (HAdd.hAdd (Neg.neg x) z …
  -/
  refine ⟨b, a, hb, ha, add_comm a b ▸ hab, ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y z : E
    h : SameRay 𝕜 (HSub.hSub x y) (HSub.hSub z x)
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    hxy : Eq (HSub.hSub x y) (HSMul.hSMul a (HSub.hSub z y))
    hzx : Eq (HSub.hSub z x) (HSMul.hSMul b (HSub.hSub z y))
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul b (HAdd.hAdd (Neg.neg x) y)) (HSMul.hSMul a (HAdd …
  -/
  rw [← sub_eq_neg_add, ← neg_sub, hxy, ← sub_eq_neg_add, hzx, smul_neg, smul_comm, neg_add_cancel]
  /-
    🎉 no goals
  -/


/-- If `z = lineMap x y c` is a point on the line passing through `x` and `y`, then the open
segment `openSegment 𝕜 x y` is included in the union of the open segments `openSegment 𝕜 x z`,
`openSegment 𝕜 z y`, and the point `z`. Informally, `(x, y) ⊆ {z} ∪ (x, z) ∪ (z, y)`. -/
theorem openSegment_subset_union (x y : E) {z : E} (hz : z ∈ range (lineMap x y : 𝕜 → E)) :
    openSegment 𝕜 x y ⊆ insert z (openSegment 𝕜 x z ∪ openSegment 𝕜 z y) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y z : E
    hz : Membership.mem (Set.range ⇑(AffineMap.lineMap x y)) z
    ⊢ HasSubset.Subset (openSegment 𝕜 x y) (Insert.insert z (Union.union (openSegm …
  -/
  rcases hz with ⟨c, rfl⟩
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y : E
    c : 𝕜
    ⊢ HasSubset.Subset (openSegment 𝕜 x y) (Insert.insert ((AffineMap.lineMap x y) …
  -/
  simp only [openSegment_eq_image_lineMap, ← mapsTo']
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y : E
    c : 𝕜
    ⊢ Set.MapsTo (⇑(AffineMap.lineMap x y)) (Set.Ioo 0 1) (Insert.insert ((AffineM …
  -/
  rintro a ⟨h₀, h₁⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y : E
    c a : 𝕜
    h₀ : LT.lt 0 a
    h₁ : LT.lt a 1
    ⊢ Membership.mem (Insert.insert ((AffineMap.lineMap x y) c) (Union.union (Set. …
  -/
  rcases lt_trichotomy a c with (hac | rfl | hca)
    /-
      case intro.intro.inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      x y : E
      c a : 𝕜
      h₀ : LT.lt 0 a
      h₁ : LT.lt a 1
      hac : LT.lt a c
      ⊢ Membership.mem (Insert.insert ((AffineMap.lineMap x y) c) (Union.union (Set. …
    -/
  · right
    /-
      case intro.intro.inl.h
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      x y : E
      c a : 𝕜
      h₀ : LT.lt 0 a
      h₁ : LT.lt a 1
      hac : LT.lt a c
      ⊢ Membership.mem (Union.union (Set.image (⇑(AffineMap.lineMap x ((AffineMap.li …
    -/
    left
    /-
      case intro.intro.inl.h.h
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      x y : E
      c a : 𝕜
      h₀ : LT.lt 0 a
      h₁ : LT.lt a 1
      hac : LT.lt a c
      ⊢ Membership.mem (Set.image (⇑(AffineMap.lineMap x ((AffineMap.lineMap x y) c) …
    -/
    have hc : 0 < c := h₀.trans hac
    /-
      case intro.intro.inl.h.h
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      x y : E
      c a : 𝕜
      h₀ : LT.lt 0 a
      h₁ : LT.lt a 1
      hac : LT.lt a c
      hc : LT.lt 0 c
      ⊢ Membership.mem (Set.image (⇑(AffineMap.lineMap x ((AffineMap.lineMap x y) c) …
    -/
    refine ⟨a / c, ⟨div_pos h₀ hc, (div_lt_one hc).2 hac⟩, ?_⟩
    /-
      case intro.intro.inl.h.h
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      x y : E
      c a : 𝕜
      h₀ : LT.lt 0 a
      h₁ : LT.lt a 1
      hac : LT.lt a c
      hc : LT.lt 0 c
      ⊢ Eq ((AffineMap.lineMap x ((AffineMap.lineMap x y) c)) (HDiv.hDiv a c)) ((Aff …
    -/
    simp only [← homothety_eq_lineMap, ← homothety_mul_apply, div_mul_cancel₀ _ hc.ne']
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr.inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      x y : E
      a : 𝕜
      h₀ : LT.lt 0 a
      h₁ : LT.lt a 1
      ⊢ Membership.mem (Insert.insert ((AffineMap.lineMap x y) a) (Union.union (Set. …
    -/
  · left
    /-
      case intro.intro.inr.inl.h
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      x y : E
      a : 𝕜
      h₀ : LT.lt 0 a
      h₁ : LT.lt a 1
      ⊢ Eq ((AffineMap.lineMap x y) a) ((AffineMap.lineMap x y) a)
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr.inr
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      x y : E
      c a : 𝕜
      h₀ : LT.lt 0 a
      h₁ : LT.lt a 1
      hca : LT.lt c a
      ⊢ Membership.mem (Insert.insert ((AffineMap.lineMap x y) c) (Union.union (Set. …
    -/
  · right
    /-
      case intro.intro.inr.inr.h
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      x y : E
      c a : 𝕜
      h₀ : LT.lt 0 a
      h₁ : LT.lt a 1
      hca : LT.lt c a
      ⊢ Membership.mem (Union.union (Set.image (⇑(AffineMap.lineMap x ((AffineMap.li …
    -/
    right
    /-
      case intro.intro.inr.inr.h.h
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      x y : E
      c a : 𝕜
      h₀ : LT.lt 0 a
      h₁ : LT.lt a 1
      hca : LT.lt c a
      ⊢ Membership.mem (Set.image (⇑(AffineMap.lineMap ((AffineMap.lineMap x y) c) y …
    -/
    have hc : 0 < 1 - c := sub_pos.2 (hca.trans h₁)
    /-
      case intro.intro.inr.inr.h.h
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      x y : E
      c a : 𝕜
      h₀ : LT.lt 0 a
      h₁ : LT.lt a 1
      hca : LT.lt c a
      hc : LT.lt 0 (HSub.hSub 1 c)
      ⊢ Membership.mem (Set.image (⇑(AffineMap.lineMap ((AffineMap.lineMap x y) c) y …
    -/
    simp only [← lineMap_apply_one_sub y]
    refine
      ⟨(a - c) / (1 - c), ⟨div_pos (sub_pos.2 hca) hc, (div_lt_one hc).2 <| sub_lt_sub_right h₁ _⟩,
        ?_⟩
    simp only [← homothety_eq_lineMap, ← homothety_mul_apply, sub_mul, one_mul,
      div_mul_cancel₀ _ hc.ne', sub_sub_sub_cancel_right]


theorem segment_subset_Icc (h : x ≤ y) : [x -[𝕜] y] ⊆ Icc x y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : OrderedAddCommMonoid E
    inst✝¹ : Module 𝕜 E
    inst✝ : OrderedSMul 𝕜 E
    x y : E
    h : LE.le x y
    ⊢ HasSubset.Subset (segment 𝕜 x y) (Set.Icc x y)
  -/
  rintro z ⟨a, b, ha, hb, hab, rfl⟩
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : OrderedAddCommMonoid E
    inst✝¹ : Module 𝕜 E
    inst✝ : OrderedSMul 𝕜 E
    x y : E
    h : LE.le x y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (Set.Icc x y) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))
  -/
  constructor
  · calc
      x = a • x + b • x := (Convex.combo_self hab _).symm
      _ ≤ a • x + b • y := by gcongr
  · calc
      a • x + b • y ≤ a • y + b • y := by gcongr
      _ = y := Convex.combo_self hab _


theorem openSegment_subset_Ioo (h : x < y) : openSegment 𝕜 x y ⊆ Ioo x y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : OrderedCancelAddCommMonoid E
    inst✝¹ : Module 𝕜 E
    inst✝ : OrderedSMul 𝕜 E
    x y : E
    h : LT.lt x y
    ⊢ HasSubset.Subset (openSegment 𝕜 x y) (Set.Ioo x y)
  -/
  rintro z ⟨a, b, ha, hb, hab, rfl⟩
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : OrderedCancelAddCommMonoid E
    inst✝¹ : Module 𝕜 E
    inst✝ : OrderedSMul 𝕜 E
    x y : E
    h : LT.lt x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (Set.Ioo x y) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))
  -/
  constructor
  · calc
      x = a • x + b • x := (Convex.combo_self hab _).symm
      _ < a • x + b • y := by gcongr
  · calc
      a • x + b • y < a • y + b • y := by gcongr
      _ = y := Convex.combo_self hab _


theorem segment_subset_uIcc (x y : E) : [x -[𝕜] y] ⊆ uIcc x y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : LinearOrderedAddCommMonoid E
    inst✝¹ : Module 𝕜 E
    inst✝ : OrderedSMul 𝕜 E
    x y : E
    ⊢ HasSubset.Subset (segment 𝕜 x y) (Set.uIcc x y)
  -/
  rcases le_total x y with h | h
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : OrderedSemiring 𝕜
      inst✝² : LinearOrderedAddCommMonoid E
      inst✝¹ : Module 𝕜 E
      inst✝ : OrderedSMul 𝕜 E
      x y : E
      h : LE.le x y
      ⊢ HasSubset.Subset (segment 𝕜 x y) (Set.uIcc x y)
    -/
  · rw [uIcc_of_le h]
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : OrderedSemiring 𝕜
      inst✝² : LinearOrderedAddCommMonoid E
      inst✝¹ : Module 𝕜 E
      inst✝ : OrderedSMul 𝕜 E
      x y : E
      h : LE.le x y
      ⊢ HasSubset.Subset (segment 𝕜 x y) (Set.Icc x y)
    -/
    exact segment_subset_Icc h
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : OrderedSemiring 𝕜
      inst✝² : LinearOrderedAddCommMonoid E
      inst✝¹ : Module 𝕜 E
      inst✝ : OrderedSMul 𝕜 E
      x y : E
      h : LE.le y x
      ⊢ HasSubset.Subset (segment 𝕜 x y) (Set.uIcc x y)
    -/
  · rw [uIcc_of_ge h, segment_symm]
    /-
      case inr
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : OrderedSemiring 𝕜
      inst✝² : LinearOrderedAddCommMonoid E
      inst✝¹ : Module 𝕜 E
      inst✝ : OrderedSMul 𝕜 E
      x y : E
      h : LE.le y x
      ⊢ HasSubset.Subset (segment 𝕜 y x) (Set.Icc y x)
    -/
    exact segment_subset_Icc h
    /-
      🎉 no goals
    -/


theorem Convex.min_le_combo (x y : E) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    min x y ≤ a • x + b • y :=
  (segment_subset_uIcc x y ⟨_, _, ha, hb, hab, rfl⟩).1


theorem Convex.combo_le_max (x y : E) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    a • x + b • y ≤ max x y :=
  (segment_subset_uIcc x y ⟨_, _, ha, hb, hab, rfl⟩).2


theorem Icc_subset_segment : Icc x y ⊆ [x -[𝕜] y] := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    x y : 𝕜
    ⊢ HasSubset.Subset (Set.Icc x y) (segment 𝕜 x y)
  -/
  rintro z ⟨hxz, hyz⟩
  /-
    case intro
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    x y z : 𝕜
    hxz : LE.le x z
    hyz : LE.le z y
    ⊢ Membership.mem (segment 𝕜 x y) z
  -/
  obtain rfl | h := (hxz.trans hyz).eq_or_lt
    /-
      case intro.inl
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      x z : 𝕜
      hxz : LE.le x z
      hyz : LE.le z x
      ⊢ Membership.mem (segment 𝕜 x x) z
    -/
  · rw [segment_same]
    /-
      case intro.inl
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      x z : 𝕜
      hxz : LE.le x z
      hyz : LE.le z x
      ⊢ Membership.mem (Singleton.singleton x) z
    -/
    exact hyz.antisymm hxz
    /-
      🎉 no goals
    -/
  /-
    case intro.inr
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    x y z : 𝕜
    hxz : LE.le x z
    hyz : LE.le z y
    h : LT.lt x y
    ⊢ Membership.mem (segment 𝕜 x y) z
  -/
  rw [← sub_nonneg] at hxz hyz
  /-
    case intro.inr
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    x y z : 𝕜
    hxz : LE.le 0 (HSub.hSub z x)
    hyz : LE.le 0 (HSub.hSub y z)
    h : LT.lt x y
    ⊢ Membership.mem (segment 𝕜 x y) z
  -/
  rw [← sub_pos] at h
  /-
    case intro.inr
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    x y z : 𝕜
    hxz : LE.le 0 (HSub.hSub z x)
    hyz : LE.le 0 (HSub.hSub y z)
    h : LT.lt 0 (HSub.hSub y x)
    ⊢ Membership.mem (segment 𝕜 x y) z
  -/
  refine ⟨(y - z) / (y - x), (z - x) / (y - x), div_nonneg hyz h.le, div_nonneg hxz h.le, ?_, ?_⟩
    /-
      case intro.inr.refine_1
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      x y z : 𝕜
      hxz : LE.le 0 (HSub.hSub z x)
      hyz : LE.le 0 (HSub.hSub y z)
      h : LT.lt 0 (HSub.hSub y x)
      ⊢ Eq (HAdd.hAdd (HDiv.hDiv (HSub.hSub y z) (HSub.hSub y x)) (HDiv.hDiv (HSub.h …
    -/
  · rw [← add_div, sub_add_sub_cancel, div_self h.ne']
    /-
      🎉 no goals
    -/
  · rw [smul_eq_mul, smul_eq_mul, ← mul_div_right_comm, ← mul_div_right_comm, ← add_div,
      div_eq_iff h.ne', add_comm, sub_mul, sub_mul, mul_comm x, sub_add_sub_cancel, mul_sub]


@[simp]
theorem segment_eq_Icc (h : x ≤ y) : [x -[𝕜] y] = Icc x y :=
  (segment_subset_Icc h).antisymm Icc_subset_segment


theorem Ioo_subset_openSegment : Ioo x y ⊆ openSegment 𝕜 x y := fun _ hz =>
  mem_openSegment_of_ne_left_right hz.1.ne hz.2.ne' <| Icc_subset_segment <| Ioo_subset_Icc_self hz


@[simp]
theorem openSegment_eq_Ioo (h : x < y) : openSegment 𝕜 x y = Ioo x y :=
  (openSegment_subset_Ioo h).antisymm Ioo_subset_openSegment


theorem segment_eq_Icc' (x y : 𝕜) : [x -[𝕜] y] = Icc (min x y) (max x y) := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    x y : 𝕜
    ⊢ Eq (segment 𝕜 x y) (Set.Icc (Min.min x y) (Max.max x y))
  -/
  rcases le_total x y with h | h
    /-
      case inl
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      x y : 𝕜
      h : LE.le x y
      ⊢ Eq (segment 𝕜 x y) (Set.Icc (Min.min x y) (Max.max x y))
    -/
  · rw [segment_eq_Icc h, max_eq_right h, min_eq_left h]
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      x y : 𝕜
      h : LE.le y x
      ⊢ Eq (segment 𝕜 x y) (Set.Icc (Min.min x y) (Max.max x y))
    -/
  · rw [segment_symm, segment_eq_Icc h, max_eq_left h, min_eq_right h]
    /-
      🎉 no goals
    -/


theorem openSegment_eq_Ioo' (hxy : x ≠ y) : openSegment 𝕜 x y = Ioo (min x y) (max x y) := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    x y : 𝕜
    hxy : Ne x y
    ⊢ Eq (openSegment 𝕜 x y) (Set.Ioo (Min.min x y) (Max.max x y))
  -/
  cases' hxy.lt_or_lt with h h
    /-
      case inl
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      x y : 𝕜
      hxy : Ne x y
      h : LT.lt x y
      ⊢ Eq (openSegment 𝕜 x y) (Set.Ioo (Min.min x y) (Max.max x y))
    -/
  · rw [openSegment_eq_Ioo h, max_eq_right h.le, min_eq_left h.le]
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      x y : 𝕜
      hxy : Ne x y
      h : LT.lt y x
      ⊢ Eq (openSegment 𝕜 x y) (Set.Ioo (Min.min x y) (Max.max x y))
    -/
  · rw [openSegment_symm, openSegment_eq_Ioo h, max_eq_left h.le, min_eq_right h.le]
    /-
      🎉 no goals
    -/


theorem segment_eq_uIcc (x y : 𝕜) : [x -[𝕜] y] = uIcc x y :=
  segment_eq_Icc' _ _


/-- A point is in an `Icc` iff it can be expressed as a convex combination of the endpoints. -/
theorem Convex.mem_Icc (h : x ≤ y) :
    z ∈ Icc x y ↔ ∃ a b, 0 ≤ a ∧ 0 ≤ b ∧ a + b = 1 ∧ a * x + b * y = z := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    x y z : 𝕜
    h : LE.le x y
    ⊢ Iff (Membership.mem (Set.Icc x y) z) (Exists fun a => Exists fun b => And (L …
  -/
  simp only [← segment_eq_Icc h, segment, mem_setOf_eq, smul_eq_mul, exists_and_left]
  /-
    🎉 no goals
  -/


/-- A point is in an `Ioo` iff it can be expressed as a strict convex combination of the endpoints.
-/
theorem Convex.mem_Ioo (h : x < y) :
    z ∈ Ioo x y ↔ ∃ a b, 0 < a ∧ 0 < b ∧ a + b = 1 ∧ a * x + b * y = z := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    x y z : 𝕜
    h : LT.lt x y
    ⊢ Iff (Membership.mem (Set.Ioo x y) z) (Exists fun a => Exists fun b => And (L …
  -/
  simp only [← openSegment_eq_Ioo h, openSegment, smul_eq_mul, exists_and_left, mem_setOf_eq]
  /-
    🎉 no goals
  -/


/-- A point is in an `Ioc` iff it can be expressed as a semistrict convex combination of the
endpoints. -/
theorem Convex.mem_Ioc (h : x < y) :
    z ∈ Ioc x y ↔ ∃ a b, 0 ≤ a ∧ 0 < b ∧ a + b = 1 ∧ a * x + b * y = z := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    x y z : 𝕜
    h : LT.lt x y
    ⊢ Iff (Membership.mem (Set.Ioc x y) z) (Exists fun a => Exists fun b => And (L …
  -/
  refine ⟨fun hz => ?_, ?_⟩
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      x y z : 𝕜
      h : LT.lt x y
      hz : Membership.mem (Set.Ioc x y) z
      ⊢ Exists fun a => Exists fun b => And (LE.le 0 a) (And (LT.lt 0 b) (And (Eq (H …
    -/
  · obtain ⟨a, b, ha, hb, hab, rfl⟩ := (Convex.mem_Icc h.le).1 (Ioc_subset_Icc_self hz)
    /-
      case refine_1.intro.intro.intro.intro.intro
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      x y : 𝕜
      h : LT.lt x y
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      hz : Membership.mem (Set.Ioc x y) (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y))
      ⊢ Exists fun a_1 => Exists fun b_1 => And (LE.le 0 a_1) (And (LT.lt 0 b_1) (An …
    -/
    obtain rfl | hb' := hb.eq_or_lt
      /-
        case refine_1.intro.intro.intro.intro.intro.inl
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        x y : 𝕜
        h : LT.lt x y
        a : 𝕜
        ha : LE.le 0 a
        hb : LE.le 0 0
        hab : Eq (HAdd.hAdd a 0) 1
        hz : Membership.mem (Set.Ioc x y) (HAdd.hAdd (HMul.hMul a x) (HMul.hMul 0 y))
        ⊢ Exists fun a_1 => Exists fun b => And (LE.le 0 a_1) (And (LT.lt 0 b) (And (E …
      -/
    · rw [add_zero] at hab
      /-
        case refine_1.intro.intro.intro.intro.intro.inl
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        x y : 𝕜
        h : LT.lt x y
        a : 𝕜
        ha : LE.le 0 a
        hb : LE.le 0 0
        hab : Eq a 1
        hz : Membership.mem (Set.Ioc x y) (HAdd.hAdd (HMul.hMul a x) (HMul.hMul 0 y))
        ⊢ Exists fun a_1 => Exists fun b => And (LE.le 0 a_1) (And (LT.lt 0 b) (And (E …
      -/
      rw [hab, one_mul, zero_mul, add_zero] at hz
      /-
        case refine_1.intro.intro.intro.intro.intro.inl
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        x y : 𝕜
        h : LT.lt x y
        a : 𝕜
        ha : LE.le 0 a
        hb : LE.le 0 0
        hab : Eq a 1
        hz : Membership.mem (Set.Ioc x y) x
        ⊢ Exists fun a_1 => Exists fun b => And (LE.le 0 a_1) (And (LT.lt 0 b) (And (E …
      -/
      exact (hz.1.ne rfl).elim
      /-
        🎉 no goals
      -/
      /-
        case refine_1.intro.intro.intro.intro.intro.inr
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        x y : 𝕜
        h : LT.lt x y
        a b : 𝕜
        ha : LE.le 0 a
        hb : LE.le 0 b
        hab : Eq (HAdd.hAdd a b) 1
        hz : Membership.mem (Set.Ioc x y) (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y))
        hb' : LT.lt 0 b
        ⊢ Exists fun a_1 => Exists fun b_1 => And (LE.le 0 a_1) (And (LT.lt 0 b_1) (An …
      -/
    · exact ⟨a, b, ha, hb', hab, rfl⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      x y z : 𝕜
      h : LT.lt x y
      ⊢ (Exists fun a => Exists fun b => And (LE.le 0 a) (And (LT.lt 0 b) (And (Eq ( …
    -/
  · rintro ⟨a, b, ha, hb, hab, rfl⟩
    /-
      case refine_2.intro.intro.intro.intro.intro
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      x y : 𝕜
      h : LT.lt x y
      a b : 𝕜
      ha : LE.le 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ Membership.mem (Set.Ioc x y) (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y))
    -/
    obtain rfl | ha' := ha.eq_or_lt
      /-
        case refine_2.intro.intro.intro.intro.intro.inl
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        x y : 𝕜
        h : LT.lt x y
        b : 𝕜
        hb : LT.lt 0 b
        ha : LE.le 0 0
        hab : Eq (HAdd.hAdd 0 b) 1
        ⊢ Membership.mem (Set.Ioc x y) (HAdd.hAdd (HMul.hMul 0 x) (HMul.hMul b y))
      -/
    · rw [zero_add] at hab
      /-
        case refine_2.intro.intro.intro.intro.intro.inl
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        x y : 𝕜
        h : LT.lt x y
        b : 𝕜
        hb : LT.lt 0 b
        ha : LE.le 0 0
        hab : Eq b 1
        ⊢ Membership.mem (Set.Ioc x y) (HAdd.hAdd (HMul.hMul 0 x) (HMul.hMul b y))
      -/
      rwa [hab, one_mul, zero_mul, zero_add, right_mem_Ioc]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.intro.intro.intro.intro.intro.inr
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        x y : 𝕜
        h : LT.lt x y
        a b : 𝕜
        ha : LE.le 0 a
        hb : LT.lt 0 b
        hab : Eq (HAdd.hAdd a b) 1
        ha' : LT.lt 0 a
        ⊢ Membership.mem (Set.Ioc x y) (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y))
      -/
    · exact Ioo_subset_Ioc_self ((Convex.mem_Ioo h).2 ⟨a, b, ha', hb, hab, rfl⟩)
      /-
        🎉 no goals
      -/


/-- A point is in an `Ico` iff it can be expressed as a semistrict convex combination of the
endpoints. -/
theorem Convex.mem_Ico (h : x < y) :
    z ∈ Ico x y ↔ ∃ a b, 0 < a ∧ 0 ≤ b ∧ a + b = 1 ∧ a * x + b * y = z := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    x y z : 𝕜
    h : LT.lt x y
    ⊢ Iff (Membership.mem (Set.Ico x y) z) (Exists fun a => Exists fun b => And (L …
  -/
  refine ⟨fun hz => ?_, ?_⟩
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      x y z : 𝕜
      h : LT.lt x y
      hz : Membership.mem (Set.Ico x y) z
      ⊢ Exists fun a => Exists fun b => And (LT.lt 0 a) (And (LE.le 0 b) (And (Eq (H …
    -/
  · obtain ⟨a, b, ha, hb, hab, rfl⟩ := (Convex.mem_Icc h.le).1 (Ico_subset_Icc_self hz)
    /-
      case refine_1.intro.intro.intro.intro.intro
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      x y : 𝕜
      h : LT.lt x y
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      hz : Membership.mem (Set.Ico x y) (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y))
      ⊢ Exists fun a_1 => Exists fun b_1 => And (LT.lt 0 a_1) (And (LE.le 0 b_1) (An …
    -/
    obtain rfl | ha' := ha.eq_or_lt
      /-
        case refine_1.intro.intro.intro.intro.intro.inl
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        x y : 𝕜
        h : LT.lt x y
        b : 𝕜
        hb : LE.le 0 b
        ha : LE.le 0 0
        hab : Eq (HAdd.hAdd 0 b) 1
        hz : Membership.mem (Set.Ico x y) (HAdd.hAdd (HMul.hMul 0 x) (HMul.hMul b y))
        ⊢ Exists fun a => Exists fun b_1 => And (LT.lt 0 a) (And (LE.le 0 b_1) (And (E …
      -/
    · rw [zero_add] at hab
      /-
        case refine_1.intro.intro.intro.intro.intro.inl
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        x y : 𝕜
        h : LT.lt x y
        b : 𝕜
        hb : LE.le 0 b
        ha : LE.le 0 0
        hab : Eq b 1
        hz : Membership.mem (Set.Ico x y) (HAdd.hAdd (HMul.hMul 0 x) (HMul.hMul b y))
        ⊢ Exists fun a => Exists fun b_1 => And (LT.lt 0 a) (And (LE.le 0 b_1) (And (E …
      -/
      rw [hab, one_mul, zero_mul, zero_add] at hz
      /-
        case refine_1.intro.intro.intro.intro.intro.inl
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        x y : 𝕜
        h : LT.lt x y
        b : 𝕜
        hb : LE.le 0 b
        ha : LE.le 0 0
        hab : Eq b 1
        hz : Membership.mem (Set.Ico x y) y
        ⊢ Exists fun a => Exists fun b_1 => And (LT.lt 0 a) (And (LE.le 0 b_1) (And (E …
      -/
      exact (hz.2.ne rfl).elim
      /-
        🎉 no goals
      -/
      /-
        case refine_1.intro.intro.intro.intro.intro.inr
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        x y : 𝕜
        h : LT.lt x y
        a b : 𝕜
        ha : LE.le 0 a
        hb : LE.le 0 b
        hab : Eq (HAdd.hAdd a b) 1
        hz : Membership.mem (Set.Ico x y) (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y))
        ha' : LT.lt 0 a
        ⊢ Exists fun a_1 => Exists fun b_1 => And (LT.lt 0 a_1) (And (LE.le 0 b_1) (An …
      -/
    · exact ⟨a, b, ha', hb, hab, rfl⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      x y z : 𝕜
      h : LT.lt x y
      ⊢ (Exists fun a => Exists fun b => And (LT.lt 0 a) (And (LE.le 0 b) (And (Eq ( …
    -/
  · rintro ⟨a, b, ha, hb, hab, rfl⟩
    /-
      case refine_2.intro.intro.intro.intro.intro
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      x y : 𝕜
      h : LT.lt x y
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ Membership.mem (Set.Ico x y) (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y))
    -/
    obtain rfl | hb' := hb.eq_or_lt
      /-
        case refine_2.intro.intro.intro.intro.intro.inl
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        x y : 𝕜
        h : LT.lt x y
        a : 𝕜
        ha : LT.lt 0 a
        hb : LE.le 0 0
        hab : Eq (HAdd.hAdd a 0) 1
        ⊢ Membership.mem (Set.Ico x y) (HAdd.hAdd (HMul.hMul a x) (HMul.hMul 0 y))
      -/
    · rw [add_zero] at hab
      /-
        case refine_2.intro.intro.intro.intro.intro.inl
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        x y : 𝕜
        h : LT.lt x y
        a : 𝕜
        ha : LT.lt 0 a
        hb : LE.le 0 0
        hab : Eq a 1
        ⊢ Membership.mem (Set.Ico x y) (HAdd.hAdd (HMul.hMul a x) (HMul.hMul 0 y))
      -/
      rwa [hab, one_mul, zero_mul, add_zero, left_mem_Ico]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.intro.intro.intro.intro.intro.inr
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        x y : 𝕜
        h : LT.lt x y
        a b : 𝕜
        ha : LT.lt 0 a
        hb : LE.le 0 b
        hab : Eq (HAdd.hAdd a b) 1
        hb' : LT.lt 0 b
        ⊢ Membership.mem (Set.Ico x y) (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y))
      -/
    · exact Ioo_subset_Ico_self ((Convex.mem_Ioo h).2 ⟨a, b, ha, hb', hab, rfl⟩)
      /-
        🎉 no goals
      -/


theorem segment_subset (x y : E × F) : segment 𝕜 x y ⊆ segment 𝕜 x.1 y.1 ×ˢ segment 𝕜 x.2 y.2 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x y : Prod E F
    ⊢ HasSubset.Subset (segment 𝕜 x y) (SProd.sprod (segment 𝕜 x.1 y.1) (segment 𝕜 …
  -/
  rintro z ⟨a, b, ha, hb, hab, hz⟩
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x y z : Prod E F
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    hz : Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y)) z
    ⊢ Membership.mem (SProd.sprod (segment 𝕜 x.1 y.1) (segment 𝕜 x.2 y.2)) z
  -/
  exact ⟨⟨a, b, ha, hb, hab, congr_arg Prod.fst hz⟩, a, b, ha, hb, hab, congr_arg Prod.snd hz⟩
  /-
    🎉 no goals
  -/


theorem openSegment_subset (x y : E × F) :
    openSegment 𝕜 x y ⊆ openSegment 𝕜 x.1 y.1 ×ˢ openSegment 𝕜 x.2 y.2 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x y : Prod E F
    ⊢ HasSubset.Subset (openSegment 𝕜 x y) (SProd.sprod (openSegment 𝕜 x.1 y.1) (o …
  -/
  rintro z ⟨a, b, ha, hb, hab, hz⟩
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x y z : Prod E F
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    hz : Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y)) z
    ⊢ Membership.mem (SProd.sprod (openSegment 𝕜 x.1 y.1) (openSegment 𝕜 x.2 y.2)) z
  -/
  exact ⟨⟨a, b, ha, hb, hab, congr_arg Prod.fst hz⟩, a, b, ha, hb, hab, congr_arg Prod.snd hz⟩
  /-
    🎉 no goals
  -/


theorem image_mk_segment_left (x₁ x₂ : E) (y : F) :
    (fun x => (x, y)) '' [x₁ -[𝕜] x₂] = [(x₁, y) -[𝕜] (x₂, y)] := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x₁ x₂ : E
    y : F
    ⊢ Eq (Set.image (fun x => { fst := x, snd := y }) (segment 𝕜 x₁ x₂)) (segment  …
  -/
  rw [segment_eq_image₂, segment_eq_image₂, image_image]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x₁ x₂ : E
    y : F
    ⊢ Eq (Set.image (fun x => { fst := HAdd.hAdd (HSMul.hSMul x.1 x₁) (HSMul.hSMul …
  -/
  refine EqOn.image_eq fun a ha ↦ ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x₁ x₂ : E
    y : F
    a : Prod 𝕜 𝕜
    ha : Membership.mem (setOf fun p => And (LE.le 0 p.1) (And (LE.le 0 p.2) (Eq ( …
    ⊢ Eq { fst := HAdd.hAdd (HSMul.hSMul a.1 x₁) (HSMul.hSMul a.2 x₂), snd := y }  …
  -/
  simp [Convex.combo_self ha.2.2]
  /-
    🎉 no goals
  -/


theorem image_mk_segment_right (x : E) (y₁ y₂ : F) :
    (fun y => (x, y)) '' [y₁ -[𝕜] y₂] = [(x, y₁) -[𝕜] (x, y₂)] := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x : E
    y₁ y₂ : F
    ⊢ Eq (Set.image (fun y => { fst := x, snd := y }) (segment 𝕜 y₁ y₂)) (segment  …
  -/
  rw [segment_eq_image₂, segment_eq_image₂, image_image]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x : E
    y₁ y₂ : F
    ⊢ Eq (Set.image (fun x_1 => { fst := x, snd := HAdd.hAdd (HSMul.hSMul x_1.1 y₁ …
  -/
  refine EqOn.image_eq fun a ha ↦ ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x : E
    y₁ y₂ : F
    a : Prod 𝕜 𝕜
    ha : Membership.mem (setOf fun p => And (LE.le 0 p.1) (And (LE.le 0 p.2) (Eq ( …
    ⊢ Eq { fst := x, snd := HAdd.hAdd (HSMul.hSMul a.1 y₁) (HSMul.hSMul a.2 y₂) }  …
  -/
  simp [Convex.combo_self ha.2.2]
  /-
    🎉 no goals
  -/


theorem image_mk_openSegment_left (x₁ x₂ : E) (y : F) :
    (fun x => (x, y)) '' openSegment 𝕜 x₁ x₂ = openSegment 𝕜 (x₁, y) (x₂, y) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x₁ x₂ : E
    y : F
    ⊢ Eq (Set.image (fun x => { fst := x, snd := y }) (openSegment 𝕜 x₁ x₂)) (open …
  -/
  rw [openSegment_eq_image₂, openSegment_eq_image₂, image_image]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x₁ x₂ : E
    y : F
    ⊢ Eq (Set.image (fun x => { fst := HAdd.hAdd (HSMul.hSMul x.1 x₁) (HSMul.hSMul …
  -/
  refine EqOn.image_eq fun a ha ↦ ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x₁ x₂ : E
    y : F
    a : Prod 𝕜 𝕜
    ha : Membership.mem (setOf fun p => And (LT.lt 0 p.1) (And (LT.lt 0 p.2) (Eq ( …
    ⊢ Eq { fst := HAdd.hAdd (HSMul.hSMul a.1 x₁) (HSMul.hSMul a.2 x₂), snd := y }  …
  -/
  simp [Convex.combo_self ha.2.2]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_mk_openSegment_right (x : E) (y₁ y₂ : F) :
    (fun y => (x, y)) '' openSegment 𝕜 y₁ y₂ = openSegment 𝕜 (x, y₁) (x, y₂) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x : E
    y₁ y₂ : F
    ⊢ Eq (Set.image (fun y => { fst := x, snd := y }) (openSegment 𝕜 y₁ y₂)) (open …
  -/
  rw [openSegment_eq_image₂, openSegment_eq_image₂, image_image]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x : E
    y₁ y₂ : F
    ⊢ Eq (Set.image (fun x_1 => { fst := x, snd := HAdd.hAdd (HSMul.hSMul x_1.1 y₁ …
  -/
  refine EqOn.image_eq fun a ha ↦ ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x : E
    y₁ y₂ : F
    a : Prod 𝕜 𝕜
    ha : Membership.mem (setOf fun p => And (LT.lt 0 p.1) (And (LT.lt 0 p.2) (Eq ( …
    ⊢ Eq { fst := x, snd := HAdd.hAdd (HSMul.hSMul a.1 y₁) (HSMul.hSMul a.2 y₂) }  …
  -/
  simp [Convex.combo_self ha.2.2]
  /-
    🎉 no goals
  -/


theorem segment_subset (x y : ∀ i, π i) : segment 𝕜 x y ⊆ s.pi fun i => segment 𝕜 (x i) (y i) := by
  /-
    𝕜 : Type u_1
    ι : Type u_5
    π : ι → Type u_6
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : (i : ι) → AddCommMonoid (π i)
    inst✝ : (i : ι) → Module 𝕜 (π i)
    s : Set ι
    x y : (i : ι) → π i
    ⊢ HasSubset.Subset (segment 𝕜 x y) (s.pi fun i => segment 𝕜 (x i) (y i))
  -/
  rintro z ⟨a, b, ha, hb, hab, hz⟩ i -
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    ι : Type u_5
    π : ι → Type u_6
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : (i : ι) → AddCommMonoid (π i)
    inst✝ : (i : ι) → Module 𝕜 (π i)
    s : Set ι
    x y z : (i : ι) → π i
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    hz : Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y)) z
    i : ι
    ⊢ Membership.mem ((fun i => segment 𝕜 (x i) (y i)) i) (z i)
  -/
  exact ⟨a, b, ha, hb, hab, congr_fun hz i⟩
  /-
    🎉 no goals
  -/


theorem openSegment_subset (x y : ∀ i, π i) :
    openSegment 𝕜 x y ⊆ s.pi fun i => openSegment 𝕜 (x i) (y i) := by
  /-
    𝕜 : Type u_1
    ι : Type u_5
    π : ι → Type u_6
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : (i : ι) → AddCommMonoid (π i)
    inst✝ : (i : ι) → Module 𝕜 (π i)
    s : Set ι
    x y : (i : ι) → π i
    ⊢ HasSubset.Subset (openSegment 𝕜 x y) (s.pi fun i => openSegment 𝕜 (x i) (y i))
  -/
  rintro z ⟨a, b, ha, hb, hab, hz⟩ i -
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    ι : Type u_5
    π : ι → Type u_6
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : (i : ι) → AddCommMonoid (π i)
    inst✝ : (i : ι) → Module 𝕜 (π i)
    s : Set ι
    x y z : (i : ι) → π i
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    hz : Eq (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y)) z
    i : ι
    ⊢ Membership.mem ((fun i => openSegment 𝕜 (x i) (y i)) i) (z i)
  -/
  exact ⟨a, b, ha, hb, hab, congr_fun hz i⟩
  /-
    🎉 no goals
  -/


theorem image_update_segment (i : ι) (x₁ x₂ : π i) (y : ∀ i, π i) :
    update y i '' [x₁ -[𝕜] x₂] = [update y i x₁ -[𝕜] update y i x₂] := by
  /-
    𝕜 : Type u_1
    ι : Type u_5
    π : ι → Type u_6
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : (i : ι) → AddCommMonoid (π i)
    inst✝¹ : (i : ι) → Module 𝕜 (π i)
    inst✝ : DecidableEq ι
    i : ι
    x₁ x₂ : π i
    y : (i : ι) → π i
    ⊢ Eq (Set.image (Function.update y i) (segment 𝕜 x₁ x₂)) (segment 𝕜 (Function. …
  -/
  rw [segment_eq_image₂, segment_eq_image₂, image_image]
  /-
    𝕜 : Type u_1
    ι : Type u_5
    π : ι → Type u_6
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : (i : ι) → AddCommMonoid (π i)
    inst✝¹ : (i : ι) → Module 𝕜 (π i)
    inst✝ : DecidableEq ι
    i : ι
    x₁ x₂ : π i
    y : (i : ι) → π i
    ⊢ Eq (Set.image (fun x => Function.update y i (HAdd.hAdd (HSMul.hSMul x.1 x₁)  …
  -/
  refine EqOn.image_eq fun a ha ↦ ?_
  /-
    𝕜 : Type u_1
    ι : Type u_5
    π : ι → Type u_6
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : (i : ι) → AddCommMonoid (π i)
    inst✝¹ : (i : ι) → Module 𝕜 (π i)
    inst✝ : DecidableEq ι
    i : ι
    x₁ x₂ : π i
    y : (i : ι) → π i
    a : Prod 𝕜 𝕜
    ha : Membership.mem (setOf fun p => And (LE.le 0 p.1) (And (LE.le 0 p.2) (Eq ( …
    ⊢ Eq (Function.update y i (HAdd.hAdd (HSMul.hSMul a.1 x₁) (HSMul.hSMul a.2 x₂) …
  -/
  simp only [← update_smul, ← update_add, Convex.combo_self ha.2.2]
  /-
    🎉 no goals
  -/


theorem image_update_openSegment (i : ι) (x₁ x₂ : π i) (y : ∀ i, π i) :
    update y i '' openSegment 𝕜 x₁ x₂ = openSegment 𝕜 (update y i x₁) (update y i x₂) := by
  /-
    𝕜 : Type u_1
    ι : Type u_5
    π : ι → Type u_6
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : (i : ι) → AddCommMonoid (π i)
    inst✝¹ : (i : ι) → Module 𝕜 (π i)
    inst✝ : DecidableEq ι
    i : ι
    x₁ x₂ : π i
    y : (i : ι) → π i
    ⊢ Eq (Set.image (Function.update y i) (openSegment 𝕜 x₁ x₂)) (openSegment 𝕜 (F …
  -/
  rw [openSegment_eq_image₂, openSegment_eq_image₂, image_image]
  /-
    𝕜 : Type u_1
    ι : Type u_5
    π : ι → Type u_6
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : (i : ι) → AddCommMonoid (π i)
    inst✝¹ : (i : ι) → Module 𝕜 (π i)
    inst✝ : DecidableEq ι
    i : ι
    x₁ x₂ : π i
    y : (i : ι) → π i
    ⊢ Eq (Set.image (fun x => Function.update y i (HAdd.hAdd (HSMul.hSMul x.1 x₁)  …
  -/
  refine EqOn.image_eq fun a ha ↦ ?_
  /-
    𝕜 : Type u_1
    ι : Type u_5
    π : ι → Type u_6
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : (i : ι) → AddCommMonoid (π i)
    inst✝¹ : (i : ι) → Module 𝕜 (π i)
    inst✝ : DecidableEq ι
    i : ι
    x₁ x₂ : π i
    y : (i : ι) → π i
    a : Prod 𝕜 𝕜
    ha : Membership.mem (setOf fun p => And (LT.lt 0 p.1) (And (LT.lt 0 p.2) (Eq ( …
    ⊢ Eq (Function.update y i (HAdd.hAdd (HSMul.hSMul a.1 x₁) (HSMul.hSMul a.2 x₂) …
  -/
  simp only [← update_smul, ← update_add, Convex.combo_self ha.2.2]
  /-
    🎉 no goals
  -/


