/-- The double coset as an element of `Set α` corresponding to `s a t` -/
def doset (a : α) (s t : Set α) : Set α :=
  s * {a} * t


lemma doset_eq_image2 (a : α) (s t : Set α) : doset a s t = Set.image2 (· * a * ·) s t := by
  /-
    α : Type u_2
    inst✝ : Mul α
    a : α
    s t : Set α
    ⊢ Eq (Doset.doset a s t) (Set.image2 (fun x1 x2 => HMul.hMul (HMul.hMul x1 a)  …
  -/
  simp_rw [doset, Set.mul_singleton, ← Set.image2_mul, Set.image2_image_left]
  /-
    🎉 no goals
  -/


theorem mem_doset {s t : Set α} {a b : α} : b ∈ doset a s t ↔ ∃ x ∈ s, ∃ y ∈ t, b = x * a * y := by
  /-
    α : Type u_2
    inst✝ : Mul α
    s t : Set α
    a b : α
    ⊢ Iff (Membership.mem (Doset.doset a s t) b) (Exists fun x => And (Membership. …
  -/
  simp only [doset_eq_image2, Set.mem_image2, eq_comm]
  /-
    🎉 no goals
  -/


theorem mem_doset_self (H K : Subgroup G) (a : G) : a ∈ doset a H K :=
  mem_doset.mpr ⟨1, H.one_mem, 1, K.one_mem, (one_mul a).symm.trans (mul_one (1 * a)).symm⟩


theorem doset_eq_of_mem {H K : Subgroup G} {a b : G} (hb : b ∈ doset a H K) :
    doset b H K = doset a H K := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    a b : G
    hb : Membership.mem (Doset.doset a ↑H ↑K) b
    ⊢ Eq (Doset.doset b ↑H ↑K) (Doset.doset a ↑H ↑K)
  -/
  obtain ⟨h, hh, k, hk, rfl⟩ := mem_doset.1 hb
  rw [doset, doset, ← Set.singleton_mul_singleton, ← Set.singleton_mul_singleton, mul_assoc,
    mul_assoc, Subgroup.singleton_mul_subgroup hk, ← mul_assoc, ← mul_assoc,
    Subgroup.subgroup_mul_singleton hh]


theorem mem_doset_of_not_disjoint {H K : Subgroup G} {a b : G}
    (h : ¬Disjoint (doset a H K) (doset b H K)) : b ∈ doset a H K := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    a b : G
    h : Not (Disjoint (Doset.doset a ↑H ↑K) (Doset.doset b ↑H ↑K))
    ⊢ Membership.mem (Doset.doset a ↑H ↑K) b
  -/
  rw [Set.not_disjoint_iff] at h
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    a b : G
    h : Exists fun x => And (Membership.mem (Doset.doset a ↑H ↑K) x) (Membership.m …
    ⊢ Membership.mem (Doset.doset a ↑H ↑K) b
  -/
  simp only [mem_doset] at *
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    a b : G
    h : Exists fun x => And (Exists fun x_1 => And (Membership.mem (↑H) x_1) (Exis …
    ⊢ Exists fun x => And (Membership.mem (↑H) x) (Exists fun y => And (Membership …
  -/
  obtain ⟨x, ⟨l, hl, r, hr, hrx⟩, y, hy, ⟨r', hr', rfl⟩⟩ := h
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    a b l : G
    hl : Membership.mem (↑H) l
    r : G
    hr : Membership.mem (↑K) r
    y : G
    hy : Membership.mem (↑H) y
    r' : G
    hr' : Membership.mem (↑K) r'
    hrx : Eq (HMul.hMul (HMul.hMul y b) r') (HMul.hMul (HMul.hMul l a) r)
    ⊢ Exists fun x => And (Membership.mem (↑H) x) (Exists fun y => And (Membership …
  -/
  refine ⟨y⁻¹ * l, H.mul_mem (H.inv_mem hy) hl, r * r'⁻¹, K.mul_mem hr (K.inv_mem hr'), ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    a b l : G
    hl : Membership.mem (↑H) l
    r : G
    hr : Membership.mem (↑K) r
    y : G
    hy : Membership.mem (↑H) y
    r' : G
    hr' : Membership.mem (↑K) r'
    hrx : Eq (HMul.hMul (HMul.hMul y b) r') (HMul.hMul (HMul.hMul l a) r)
    ⊢ Eq b (HMul.hMul (HMul.hMul (HMul.hMul (Inv.inv y) l) a) (HMul.hMul r (Inv.in …
  -/
  rwa [mul_assoc, mul_assoc, eq_inv_mul_iff_mul_eq, ← mul_assoc, ← mul_assoc, eq_mul_inv_iff_mul_eq]
  /-
    🎉 no goals
  -/


theorem eq_of_not_disjoint {H K : Subgroup G} {a b : G}
    (h : ¬Disjoint (doset a H K) (doset b H K)) : doset a H K = doset b H K := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    a b : G
    h : Not (Disjoint (Doset.doset a ↑H ↑K) (Doset.doset b ↑H ↑K))
    ⊢ Eq (Doset.doset a ↑H ↑K) (Doset.doset b ↑H ↑K)
  -/
  rw [disjoint_comm] at h
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    a b : G
    h : Not (Disjoint (Doset.doset b ↑H ↑K) (Doset.doset a ↑H ↑K))
    ⊢ Eq (Doset.doset a ↑H ↑K) (Doset.doset b ↑H ↑K)
  -/
  have ha : a ∈ doset b H K := mem_doset_of_not_disjoint h
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    a b : G
    h : Not (Disjoint (Doset.doset b ↑H ↑K) (Doset.doset a ↑H ↑K))
    ha : Membership.mem (Doset.doset b ↑H ↑K) a
    ⊢ Eq (Doset.doset a ↑H ↑K) (Doset.doset b ↑H ↑K)
  -/
  apply doset_eq_of_mem ha
  /-
    🎉 no goals
  -/


/-- The setoid defined by the double_coset relation -/
def setoid (H K : Set G) : Setoid G :=
  Setoid.ker fun x => doset x H K


/-- Quotient of `G` by the double coset relation, i.e. `H \ G / K` -/
def Quotient (H K : Set G) : Type _ :=
  _root_.Quotient (setoid H K)


theorem rel_iff {H K : Subgroup G} {x y : G} :
    setoid ↑H ↑K x y ↔ ∃ a ∈ H, ∃ b ∈ K, y = a * x * b :=
  Iff.trans
    ⟨fun (hxy : doset x H K = doset y H K) => hxy ▸ mem_doset_self H K y,
      fun hxy => (doset_eq_of_mem hxy).symm⟩ mem_doset


theorem bot_rel_eq_leftRel (H : Subgroup G) :
    ⇑(setoid ↑(⊥ : Subgroup G) ↑H) = ⇑(QuotientGroup.leftRel H) := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Eq ⇑(Doset.setoid ↑Bot.bot ↑H) ⇑(QuotientGroup.leftRel H)
  -/
  ext a b
  /-
    case h.h.a
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    a b : G
    ⊢ Iff ((Doset.setoid ↑Bot.bot ↑H) a b) ((QuotientGroup.leftRel H) a b)
  -/
  rw [rel_iff, QuotientGroup.leftRel_apply]
  /-
    case h.h.a
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    a b : G
    ⊢ Iff (Exists fun a_1 => And (Membership.mem Bot.bot a_1) (Exists fun b_1 => A …
  -/
  constructor
    /-
      case h.h.a.mp
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      a b : G
      ⊢ (Exists fun a_1 => And (Membership.mem Bot.bot a_1) (Exists fun b_1 => And ( …
    -/
  · rintro ⟨a, rfl : a = 1, b, hb, rfl⟩
    /-
      case h.h.a.mp.intro.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      a b : G
      hb : Membership.mem H b
      ⊢ Membership.mem H (HMul.hMul (Inv.inv a) (HMul.hMul (HMul.hMul 1 a) b))
    -/
    rwa [one_mul, inv_mul_cancel_left]
    /-
      🎉 no goals
    -/
    /-
      case h.h.a.mpr
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      a b : G
      ⊢ Membership.mem H (HMul.hMul (Inv.inv a) b) → Exists fun a_2 => And (Membersh …
    -/
  · rintro (h : a⁻¹ * b ∈ H)
    /-
      case h.h.a.mpr
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      a b : G
      h : Membership.mem H (HMul.hMul (Inv.inv a) b)
      ⊢ Exists fun a_1 => And (Membership.mem Bot.bot a_1) (Exists fun b_1 => And (M …
    -/
    exact ⟨1, rfl, a⁻¹ * b, h, by rw [one_mul, mul_inv_cancel_left]⟩
    /-
      🎉 no goals
    -/


theorem rel_bot_eq_right_group_rel (H : Subgroup G) :
    ⇑(setoid ↑H ↑(⊥ : Subgroup G)) = ⇑(QuotientGroup.rightRel H) := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Eq ⇑(Doset.setoid ↑H ↑Bot.bot) ⇑(QuotientGroup.rightRel H)
  -/
  ext a b
  /-
    case h.h.a
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    a b : G
    ⊢ Iff ((Doset.setoid ↑H ↑Bot.bot) a b) ((QuotientGroup.rightRel H) a b)
  -/
  rw [rel_iff, QuotientGroup.rightRel_apply]
  /-
    case h.h.a
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    a b : G
    ⊢ Iff (Exists fun a_1 => And (Membership.mem H a_1) (Exists fun b_1 => And (Me …
  -/
  constructor
    /-
      case h.h.a.mp
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      a b : G
      ⊢ (Exists fun a_1 => And (Membership.mem H a_1) (Exists fun b_1 => And (Member …
    -/
  · rintro ⟨b, hb, a, rfl : a = 1, rfl⟩
    /-
      case h.h.a.mp.intro.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      a b : G
      hb : Membership.mem H b
      ⊢ Membership.mem H (HMul.hMul (HMul.hMul (HMul.hMul b a) 1) (Inv.inv a))
    -/
    rwa [mul_one, mul_inv_cancel_right]
    /-
      🎉 no goals
    -/
    /-
      case h.h.a.mpr
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      a b : G
      ⊢ Membership.mem H (HMul.hMul b (Inv.inv a)) → Exists fun a_2 => And (Membersh …
    -/
  · rintro (h : b * a⁻¹ ∈ H)
    /-
      case h.h.a.mpr
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      a b : G
      h : Membership.mem H (HMul.hMul b (Inv.inv a))
      ⊢ Exists fun a_1 => And (Membership.mem H a_1) (Exists fun b_1 => And (Members …
    -/
    exact ⟨b * a⁻¹, h, 1, rfl, by rw [mul_one, inv_mul_cancel_right]⟩
    /-
      🎉 no goals
    -/


/-- Create a doset out of an element of `H \ G / K`-/
def quotToDoset (H K : Subgroup G) (q : Quotient (H : Set G) K) : Set G :=
  doset q.out H K


/-- Map from `G` to `H \ G / K`-/
abbrev mk (H K : Subgroup G) (a : G) : Quotient (H : Set G) K :=
  Quotient.mk'' a


instance (H K : Subgroup G) : Inhabited (Quotient (H : Set G) K) :=
  ⟨mk H K (1 : G)⟩


theorem eq (H K : Subgroup G) (a b : G) :
    mk H K a = mk H K b ↔ ∃ h ∈ H, ∃ k ∈ K, b = h * a * k := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    a b : G
    ⊢ Iff (Eq (Doset.mk H K a) (Doset.mk H K b)) (Exists fun h => And (Membership. …
  -/
  rw [Quotient.eq'']
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    a b : G
    ⊢ Iff ((Doset.setoid ↑H ↑K) a b) (Exists fun h => And (Membership.mem H h) (Ex …
  -/
  apply rel_iff
  /-
    🎉 no goals
  -/


theorem out_eq' (H K : Subgroup G) (q : Quotient ↑H ↑K) : mk H K q.out = q :=
  Quotient.out_eq' q


theorem mk_out_eq_mul (H K : Subgroup G) (g : G) :
    ∃ h k : G, h ∈ H ∧ k ∈ K ∧ (mk H K g : Quotient ↑H ↑K).out = h * g * k := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    g : G
    ⊢ Exists fun h => Exists fun k => And (Membership.mem H h) (And (Membership.me …
  -/
  have := eq H K (mk H K g : Quotient ↑H ↑K).out g
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    g : G
    this : Iff (Eq (Doset.mk H K (Quotient.out (Doset.mk H K g))) (Doset.mk H K g) …
    ⊢ Exists fun h => Exists fun k => And (Membership.mem H h) (And (Membership.me …
  -/
  rw [out_eq'] at this
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    g : G
    this : Iff (Eq (Doset.mk H K g) (Doset.mk H K g)) (Exists fun h => And (Member …
    ⊢ Exists fun h => Exists fun k => And (Membership.mem H h) (And (Membership.me …
  -/
  obtain ⟨h, h_h, k, hk, T⟩ := this.1 rfl
  /-
    case intro.intro.intro.intro
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    g : G
    this : Iff (Eq (Doset.mk H K g) (Doset.mk H K g)) (Exists fun h => And (Member …
    h : G
    h_h : Membership.mem H h
    k : G
    hk : Membership.mem K k
    T : Eq g (HMul.hMul (HMul.hMul h (Quotient.out (Doset.mk H K g))) k)
    ⊢ Exists fun h => Exists fun k => And (Membership.mem H h) (And (Membership.me …
  -/
  refine ⟨h⁻¹, k⁻¹, H.inv_mem h_h, K.inv_mem hk, eq_mul_inv_of_mul_eq (eq_inv_mul_of_mul_eq ?_)⟩
  /-
    case intro.intro.intro.intro
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    g : G
    this : Iff (Eq (Doset.mk H K g) (Doset.mk H K g)) (Exists fun h => And (Member …
    h : G
    h_h : Membership.mem H h
    k : G
    hk : Membership.mem K k
    T : Eq g (HMul.hMul (HMul.hMul h (Quotient.out (Doset.mk H K g))) k)
    ⊢ Eq (HMul.hMul h (HMul.hMul (Quotient.out (Doset.mk H K g)) k)) g
  -/
  rw [← mul_assoc, ← T]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-19")] alias mk_out'_eq_mul := mk_out_eq_mul


theorem mk_eq_of_doset_eq {H K : Subgroup G} {a b : G} (h : doset a H K = doset b H K) :
    mk H K a = mk H K b := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    a b : G
    h : Eq (Doset.doset a ↑H ↑K) (Doset.doset b ↑H ↑K)
    ⊢ Eq (Doset.mk H K a) (Doset.mk H K b)
  -/
  rw [eq]
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    a b : G
    h : Eq (Doset.doset a ↑H ↑K) (Doset.doset b ↑H ↑K)
    ⊢ Exists fun h => And (Membership.mem H h) (Exists fun k => And (Membership.me …
  -/
  exact mem_doset.mp (h.symm ▸ mem_doset_self H K b)
  /-
    🎉 no goals
  -/


theorem disjoint_out {H K : Subgroup G} {a b : Quotient H K} :
    a ≠ b → Disjoint (doset a.out H K) (doset b.out (H : Set G) K) := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    a b : Doset.Quotient ↑H ↑K
    ⊢ Ne a b → Disjoint (Doset.doset (Quotient.out a) ↑H ↑K) (Doset.doset (Quotien …
  -/
  contrapose!
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    a b : Doset.Quotient ↑H ↑K
    ⊢ Not (Disjoint (Doset.doset (Quotient.out a) ↑H ↑K) (Doset.doset (Quotient.ou …
  -/
  intro h
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    a b : Doset.Quotient ↑H ↑K
    h : Not (Disjoint (Doset.doset (Quotient.out a) ↑H ↑K) (Doset.doset (Quotient. …
    ⊢ Eq a b
  -/
  simpa [out_eq'] using mk_eq_of_doset_eq (eq_of_not_disjoint h)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-19")] alias disjoint_out' := disjoint_out


theorem union_quotToDoset (H K : Subgroup G) : ⋃ q, quotToDoset H K q = Set.univ := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    ⊢ Eq (Set.iUnion fun q => Doset.quotToDoset H K q) Set.univ
  -/
  ext x
  simp only [Set.mem_iUnion, quotToDoset, mem_doset, SetLike.mem_coe, exists_prop, Set.mem_univ,
    iff_true]
  /-
    case h
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    x : G
    ⊢ Exists fun i => Exists fun x_1 => And (Membership.mem H x_1) (Exists fun y = …
  -/
  use mk H K x
  /-
    case h
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    x : G
    ⊢ Exists fun x_1 => And (Membership.mem H x_1) (Exists fun y => And (Membershi …
  -/
  obtain ⟨h, k, h3, h4, h5⟩ := mk_out_eq_mul H K x
  /-
    case h.intro.intro.intro.intro
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    x h k : G
    h3 : Membership.mem H h
    h4 : Membership.mem K k
    h5 : Eq (Quotient.out (Doset.mk H K x)) (HMul.hMul (HMul.hMul h x) k)
    ⊢ Exists fun x_1 => And (Membership.mem H x_1) (Exists fun y => And (Membershi …
  -/
  refine ⟨h⁻¹, H.inv_mem h3, k⁻¹, K.inv_mem h4, ?_⟩
  /-
    case h.intro.intro.intro.intro
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    x h k : G
    h3 : Membership.mem H h
    h4 : Membership.mem K k
    h5 : Eq (Quotient.out (Doset.mk H K x)) (HMul.hMul (HMul.hMul h x) k)
    ⊢ Eq x (HMul.hMul (HMul.hMul (Inv.inv h) (Quotient.out (Doset.mk H K x))) (Inv …
  -/
  simp only [h5, Subgroup.coe_mk, ← mul_assoc, one_mul, inv_mul_cancel, mul_inv_cancel_right]
  /-
    🎉 no goals
  -/


theorem doset_union_rightCoset (H K : Subgroup G) (a : G) :
    ⋃ k : K, op (a * k) • ↑H = doset a H K := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    a : G
    ⊢ Eq (Set.iUnion fun k => HSMul.hSMul (MulOpposite.op (HMul.hMul a ↑k)) ↑H) (D …
  -/
  ext x
  simp only [mem_rightCoset_iff, exists_prop, mul_inv_rev, Set.mem_iUnion, mem_doset,
    Subgroup.mem_carrier, SetLike.mem_coe]
  /-
    case h
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    a x : G
    ⊢ Iff (Exists fun i => Membership.mem H (HMul.hMul x (HMul.hMul (Inv.inv ↑i) ( …
  -/
  constructor
    /-
      case h.mp
      G : Type u_1
      inst✝ : Group G
      H K : Subgroup G
      a x : G
      ⊢ (Exists fun i => Membership.mem H (HMul.hMul x (HMul.hMul (Inv.inv ↑i) (Inv. …
    -/
  · rintro ⟨y, h_h⟩
    /-
      case h.mp.intro
      G : Type u_1
      inst✝ : Group G
      H K : Subgroup G
      a x : G
      y : Subtype fun x => Membership.mem K x
      h_h : Membership.mem H (HMul.hMul x (HMul.hMul (Inv.inv ↑y) (Inv.inv a)))
      ⊢ Exists fun x_1 => And (Membership.mem H x_1) (Exists fun y => And (Membershi …
    -/
    refine ⟨x * (y⁻¹ * a⁻¹), h_h, y, y.2, ?_⟩
    /-
      case h.mp.intro
      G : Type u_1
      inst✝ : Group G
      H K : Subgroup G
      a x : G
      y : Subtype fun x => Membership.mem K x
      h_h : Membership.mem H (HMul.hMul x (HMul.hMul (Inv.inv ↑y) (Inv.inv a)))
      ⊢ Eq x (HMul.hMul (HMul.hMul (HMul.hMul x (HMul.hMul (↑(Inv.inv y)) (Inv.inv a …
    -/
    simp only [← mul_assoc, Subgroup.coe_mk, inv_mul_cancel_right, InvMemClass.coe_inv]
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      G : Type u_1
      inst✝ : Group G
      H K : Subgroup G
      a x : G
      ⊢ (Exists fun x_1 => And (Membership.mem H x_1) (Exists fun y => And (Membersh …
    -/
  · rintro ⟨x, hx, y, hy, hxy⟩
    /-
      case h.mpr.intro.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      H K : Subgroup G
      a x✝ x : G
      hx : Membership.mem H x
      y : G
      hy : Membership.mem K y
      hxy : Eq x✝ (HMul.hMul (HMul.hMul x a) y)
      ⊢ Exists fun i => Membership.mem H (HMul.hMul x✝ (HMul.hMul (Inv.inv ↑i) (Inv. …
    -/
    refine ⟨⟨y, hy⟩, ?_⟩
    /-
      case h.mpr.intro.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      H K : Subgroup G
      a x✝ x : G
      hx : Membership.mem H x
      y : G
      hy : Membership.mem K y
      hxy : Eq x✝ (HMul.hMul (HMul.hMul x a) y)
      ⊢ Membership.mem H (HMul.hMul x✝ (HMul.hMul (Inv.inv ↑⟨y, hy⟩) (Inv.inv a)))
    -/
    simp only [hxy, ← mul_assoc, hx, mul_inv_cancel_right, Subgroup.coe_mk]
    /-
      🎉 no goals
    -/


theorem doset_union_leftCoset (H K : Subgroup G) (a : G) :
    ⋃ h : H, (h * a : G) • ↑K = doset a H K := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    a : G
    ⊢ Eq (Set.iUnion fun h => HSMul.hSMul (HMul.hMul (↑h) a) ↑K) (Doset.doset a ↑H …
  -/
  ext x
  /-
    case h
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    a x : G
    ⊢ Iff (Membership.mem (Set.iUnion fun h => HSMul.hSMul (HMul.hMul (↑h) a) ↑K)  …
  -/
  simp only [mem_leftCoset_iff, mul_inv_rev, Set.mem_iUnion, mem_doset]
  /-
    case h
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    a x : G
    ⊢ Iff (Exists fun i => Membership.mem (↑K) (HMul.hMul (HMul.hMul (Inv.inv a) ( …
  -/
  constructor
    /-
      case h.mp
      G : Type u_1
      inst✝ : Group G
      H K : Subgroup G
      a x : G
      ⊢ (Exists fun i => Membership.mem (↑K) (HMul.hMul (HMul.hMul (Inv.inv a) (Inv. …
    -/
  · rintro ⟨y, h_h⟩
    /-
      case h.mp.intro
      G : Type u_1
      inst✝ : Group G
      H K : Subgroup G
      a x : G
      y : Subtype fun x => Membership.mem H x
      h_h : Membership.mem (↑K) (HMul.hMul (HMul.hMul (Inv.inv a) (Inv.inv ↑y)) x)
      ⊢ Exists fun x_1 => And (Membership.mem (↑H) x_1) (Exists fun y => And (Member …
    -/
    refine ⟨y, y.2, a⁻¹ * y⁻¹ * x, h_h, ?_⟩
    /-
      case h.mp.intro
      G : Type u_1
      inst✝ : Group G
      H K : Subgroup G
      a x : G
      y : Subtype fun x => Membership.mem H x
      h_h : Membership.mem (↑K) (HMul.hMul (HMul.hMul (Inv.inv a) (Inv.inv ↑y)) x)
      ⊢ Eq x (HMul.hMul (HMul.hMul (↑y) a) (HMul.hMul (HMul.hMul (Inv.inv a) ↑(Inv.i …
    -/
    simp only [← mul_assoc, one_mul, mul_inv_cancel, mul_inv_cancel_right, InvMemClass.coe_inv]
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      G : Type u_1
      inst✝ : Group G
      H K : Subgroup G
      a x : G
      ⊢ (Exists fun x_1 => And (Membership.mem (↑H) x_1) (Exists fun y => And (Membe …
    -/
  · rintro ⟨x, hx, y, hy, hxy⟩
    /-
      case h.mpr.intro.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      H K : Subgroup G
      a x✝ x : G
      hx : Membership.mem (↑H) x
      y : G
      hy : Membership.mem (↑K) y
      hxy : Eq x✝ (HMul.hMul (HMul.hMul x a) y)
      ⊢ Exists fun i => Membership.mem (↑K) (HMul.hMul (HMul.hMul (Inv.inv a) (Inv.i …
    -/
    refine ⟨⟨x, hx⟩, ?_⟩
    /-
      case h.mpr.intro.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      H K : Subgroup G
      a x✝ x : G
      hx : Membership.mem (↑H) x
      y : G
      hy : Membership.mem (↑K) y
      hxy : Eq x✝ (HMul.hMul (HMul.hMul x a) y)
      ⊢ Membership.mem (↑K) (HMul.hMul (HMul.hMul (Inv.inv a) (Inv.inv ↑⟨x, hx⟩)) x✝)
    -/
    simp only [hxy, ← mul_assoc, hy, one_mul, inv_mul_cancel, Subgroup.coe_mk, inv_mul_cancel_right]
    /-
      🎉 no goals
    -/


theorem left_bot_eq_left_quot (H : Subgroup G) :
    Quotient (⊥ : Subgroup G) (H : Set G) = (G ⧸ H) := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Eq (Doset.Quotient ↑Bot.bot ↑H) (HasQuotient.Quotient G H)
  -/
  unfold Quotient
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Eq (_root_.Quotient (Doset.setoid ↑Bot.bot ↑H)) (HasQuotient.Quotient G H)
  -/
  congr
  /-
    case e_s
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Eq (Doset.setoid ↑Bot.bot ↑H) (QuotientGroup.leftRel H)
  -/
  ext
  /-
    case e_s.a
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    a✝ b✝ : G
    ⊢ Iff ((Doset.setoid ↑Bot.bot ↑H) a✝ b✝) ((QuotientGroup.leftRel H) a✝ b✝)
  -/
  simp_rw [← bot_rel_eq_leftRel H]
  /-
    🎉 no goals
  -/


theorem right_bot_eq_right_quot (H : Subgroup G) :
    Quotient (H : Set G) (⊥ : Subgroup G) = _root_.Quotient (QuotientGroup.rightRel H) := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Eq (Doset.Quotient ↑H ↑Bot.bot) (_root_.Quotient (QuotientGroup.rightRel H))
  -/
  unfold Quotient
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Eq (_root_.Quotient (Doset.setoid ↑H ↑Bot.bot)) (_root_.Quotient (QuotientGr …
  -/
  congr
  /-
    case e_s
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Eq (Doset.setoid ↑H ↑Bot.bot) (QuotientGroup.rightRel H)
  -/
  ext
  /-
    case e_s.a
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    a✝ b✝ : G
    ⊢ Iff ((Doset.setoid ↑H ↑Bot.bot) a✝ b✝) ((QuotientGroup.rightRel H) a✝ b✝)
  -/
  simp_rw [← rel_bot_eq_right_group_rel H]
  /-
    🎉 no goals
  -/


