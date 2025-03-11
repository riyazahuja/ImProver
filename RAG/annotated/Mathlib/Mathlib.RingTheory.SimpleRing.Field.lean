open TwoSidedIdeal in
lemma isField_center (A : Type*) [Ring A] [IsSimpleRing A] : IsField (Subring.center A) where
  exists_pair_ne := ⟨0, 1, zero_ne_one⟩
  mul_comm := mul_comm
  mul_inv_cancel := by
    /-
      A : Type u_1
      inst✝¹ : Ring A
      inst✝ : IsSimpleRing A
      ⊢ ∀ {a : Subtype fun x => Membership.mem (Subring.center A) x}, Ne a 0 → Exist …
    -/
    rintro ⟨x, hx1⟩ hx2
    /-
      case mk
      A : Type u_1
      inst✝¹ : Ring A
      inst✝ : IsSimpleRing A
      x : A
      hx1 : Membership.mem (Subring.center A) x
      hx2 : Ne ⟨x, hx1⟩ 0
      ⊢ Exists fun b => Eq (HMul.hMul ⟨x, hx1⟩ b) 1
    -/
    rw [Subring.mem_center_iff] at hx1
    /-
      case mk
      A : Type u_1
      inst✝¹ : Ring A
      inst✝ : IsSimpleRing A
      x : A
      hx1✝ : Membership.mem (Subring.center A) x
      hx1 : ∀ (g : A), Eq (HMul.hMul g x) (HMul.hMul x g)
      hx2 : Ne ⟨x, hx1✝⟩ 0
      ⊢ Exists fun b => Eq (HMul.hMul ⟨x, hx1✝⟩ b) 1
    -/
    replace hx2 : x ≠ 0 := by simpa [Subtype.ext_iff] using hx2
    -- Todo: golf the following `let` once `TwoSidedIdeal.span` is defined
    let I := TwoSidedIdeal.mk' (Set.range (x * ·)) ⟨0, by simp⟩
      (by rintro _ _ ⟨x, rfl⟩ ⟨y, rfl⟩; exact ⟨x + y, mul_add _ _ _⟩)
      (by rintro _ ⟨x, rfl⟩; exact ⟨-x, by simp⟩)
      (by rintro a _ ⟨c, rfl⟩; exact ⟨a * c, by dsimp; rw [← mul_assoc, ← hx1, mul_assoc]⟩)
      (by rintro _ b ⟨a, rfl⟩; exact ⟨a * b, by dsimp; rw [← mul_assoc, ← hx1, mul_assoc]⟩)
    /-
      case mk
      A : Type u_1
      inst✝¹ : Ring A
      inst✝ : IsSimpleRing A
      x : A
      hx1✝ : Membership.mem (Subring.center A) x
      hx1 : ∀ (g : A), Eq (HMul.hMul g x) (HMul.hMul x g)
      hx2 : Ne x 0
      I : TwoSidedIdeal A := TwoSidedIdeal.mk' (Set.range fun x_1 => HMul.hMul x x_1 …
      ⊢ Exists fun b => Eq (HMul.hMul ⟨x, hx1✝⟩ b) 1
    -/
    have mem : 1 ∈ I := one_mem_of_ne_zero_mem I hx2 (by simpa [I, mem_mk'] using ⟨1, by simp⟩)
    /-
      case mk
      A : Type u_1
      inst✝¹ : Ring A
      inst✝ : IsSimpleRing A
      x : A
      hx1✝ : Membership.mem (Subring.center A) x
      hx1 : ∀ (g : A), Eq (HMul.hMul g x) (HMul.hMul x g)
      hx2 : Ne x 0
      I : TwoSidedIdeal A := TwoSidedIdeal.mk' (Set.range fun x_1 => HMul.hMul x x_1 …
      mem : Membership.mem I 1
      ⊢ Exists fun b => Eq (HMul.hMul ⟨x, hx1✝⟩ b) 1
    -/
    simp only [TwoSidedIdeal.mem_mk', Set.mem_range, I] at mem
    /-
      case mk
      A : Type u_1
      inst✝¹ : Ring A
      inst✝ : IsSimpleRing A
      x : A
      hx1✝ : Membership.mem (Subring.center A) x
      hx1 : ∀ (g : A), Eq (HMul.hMul g x) (HMul.hMul x g)
      hx2 : Ne x 0
      I : TwoSidedIdeal A := TwoSidedIdeal.mk' (Set.range fun x_1 => HMul.hMul x x_1 …
      mem : Exists fun y => Eq (HMul.hMul x y) 1
      ⊢ Exists fun b => Eq (HMul.hMul ⟨x, hx1✝⟩ b) 1
    -/
    obtain ⟨y, hy⟩ := mem
    /-
      case mk.intro
      A : Type u_1
      inst✝¹ : Ring A
      inst✝ : IsSimpleRing A
      x : A
      hx1✝ : Membership.mem (Subring.center A) x
      hx1 : ∀ (g : A), Eq (HMul.hMul g x) (HMul.hMul x g)
      hx2 : Ne x 0
      I : TwoSidedIdeal A := TwoSidedIdeal.mk' (Set.range fun x_1 => HMul.hMul x x_1 …
      y : A
      hy : Eq (HMul.hMul x y) 1
      ⊢ Exists fun b => Eq (HMul.hMul ⟨x, hx1✝⟩ b) 1
    -/
    refine ⟨⟨y, Subring.mem_center_iff.2 fun a ↦ ?_⟩, by ext; exact hy⟩
    calc a * y
      _ = (x * y) * a * y := by rw [hy, one_mul]
      _ = (y * x) * a * y := by rw [hx1]
      _ = y * (x * a) * y := by rw [mul_assoc y x a]
      _ = y * (a * x) * y := by rw [hx1]
      _ = y * ((a * x) * y) := by rw [mul_assoc]
      _ = y * (a * (x * y)) := by rw [mul_assoc a x y]
      _ = y * a := by rw [hy, mul_one]


lemma isSimpleRing_iff_isField (A : Type*) [CommRing A] : IsSimpleRing A ↔ IsField A :=
  ⟨fun _ ↦ Subring.topEquiv.symm.toMulEquiv.isField _ <| by
    /-
      A : Type u_1
      inst✝ : CommRing A
      x✝ : IsSimpleRing A
      ⊢ IsField (Subtype fun x => Membership.mem Top.top x)
    -/
    rw [← Subring.center_eq_top A]; exact IsSimpleRing.isField_center A,
                                    /-
                                      🎉 no goals
                                    -/
    fun h ↦ letI := h.toField; inferInstance⟩

