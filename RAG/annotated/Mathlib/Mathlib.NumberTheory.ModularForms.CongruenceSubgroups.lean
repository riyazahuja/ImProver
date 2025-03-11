local notation "SLMOD(" N ")" =>
  @Matrix.SpecialLinearGroup.map (Fin 2) _ _ _ _ _ _ (Int.castRingHom (ZMod N))


@[simp]
theorem SL_reduction_mod_hom_val (γ : SL(2, ℤ)) (i j : Fin 2):
    SLMOD(N) γ i j = (γ i j : ZMod N) :=
  rfl


/-- The full level `N` congruence subgroup of `SL(2, ℤ)` of matrices that reduce to the identity
modulo `N`. -/
def Gamma : Subgroup SL(2, ℤ) :=
  SLMOD(N).ker


@[inherit_doc] scoped notation  "Γ(" n ")"  => Gamma n


theorem Gamma_mem' {N} {γ : SL(2, ℤ)} : γ ∈ Gamma N ↔ SLMOD(N) γ = 1 :=
  Iff.rfl


@[simp]
theorem Gamma_mem {N} {γ : SL(2, ℤ)} : γ ∈ Gamma N ↔ (γ 0 0 : ZMod N) = 1 ∧
    (γ 0 1 : ZMod N) = 0 ∧ (γ 1 0 : ZMod N) = 0 ∧ (γ 1 1 : ZMod N) = 1 := by
  /-
    N : Nat
    γ : Matrix.SpecialLinearGroup (Fin 2) Int
    ⊢ Iff (Membership.mem (CongruenceSubgroup.Gamma N) γ) (And (Eq (↑(↑γ 0 0)) 1)  …
  -/
  rw [Gamma_mem']
  /-
    N : Nat
    γ : Matrix.SpecialLinearGroup (Fin 2) Int
    ⊢ Iff (Eq ((Matrix.SpecialLinearGroup.map (Int.castRingHom (ZMod N))) γ) 1) (A …
  -/
  constructor
    /-
      case mp
      N : Nat
      γ : Matrix.SpecialLinearGroup (Fin 2) Int
      ⊢ Eq ((Matrix.SpecialLinearGroup.map (Int.castRingHom (ZMod N))) γ) 1 → And (E …
    -/
  · intro h
    /-
      case mp
      N : Nat
      γ : Matrix.SpecialLinearGroup (Fin 2) Int
      h : Eq ((Matrix.SpecialLinearGroup.map (Int.castRingHom (ZMod N))) γ) 1
      ⊢ And (Eq (↑(↑γ 0 0)) 1) (And (Eq (↑(↑γ 0 1)) 0) (And (Eq (↑(↑γ 1 0)) 0) (Eq ( …
    -/
    simp [← SL_reduction_mod_hom_val N γ, h]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      N : Nat
      γ : Matrix.SpecialLinearGroup (Fin 2) Int
      ⊢ And (Eq (↑(↑γ 0 0)) 1) (And (Eq (↑(↑γ 0 1)) 0) (And (Eq (↑(↑γ 1 0)) 0) (Eq ( …
    -/
  · intro h
    /-
      case mpr
      N : Nat
      γ : Matrix.SpecialLinearGroup (Fin 2) Int
      h : And (Eq (↑(↑γ 0 0)) 1) (And (Eq (↑(↑γ 0 1)) 0) (And (Eq (↑(↑γ 1 0)) 0) (Eq …
      ⊢ Eq ((Matrix.SpecialLinearGroup.map (Int.castRingHom (ZMod N))) γ) 1
    -/
    ext i j
    /-
      case mpr.a
      N : Nat
      γ : Matrix.SpecialLinearGroup (Fin 2) Int
      h : And (Eq (↑(↑γ 0 0)) 1) (And (Eq (↑(↑γ 0 1)) 0) (And (Eq (↑(↑γ 1 0)) 0) (Eq …
      i j : Fin 2
      ⊢ Eq (↑((Matrix.SpecialLinearGroup.map (Int.castRingHom (ZMod N))) γ) i j) (↑1 …
    -/
    rw [SL_reduction_mod_hom_val N γ]
    /-
      case mpr.a
      N : Nat
      γ : Matrix.SpecialLinearGroup (Fin 2) Int
      h : And (Eq (↑(↑γ 0 0)) 1) (And (Eq (↑(↑γ 0 1)) 0) (And (Eq (↑(↑γ 1 0)) 0) (Eq …
      i j : Fin 2
      ⊢ Eq (↑(↑γ i j)) (↑1 i j)
    -/
    fin_cases i <;> fin_cases j <;> simp only [h]
    /-
      case mpr.a.«0».«0»
      N : Nat
      γ : Matrix.SpecialLinearGroup (Fin 2) Int
      h : And (Eq (↑(↑γ 0 0)) 1) (And (Eq (↑(↑γ 0 1)) 0) (And (Eq (↑(↑γ 1 0)) 0) (Eq …
      ⊢ Eq (↑(↑γ ⟨0, ⋯⟩ ⟨0, ⋯⟩)) (↑1 ⟨0, ⋯⟩ ⟨0, ⋯⟩)
    -/
    exacts [h.1, h.2.1, h.2.2.1, h.2.2.2]
    /-
      🎉 no goals
    -/


theorem Gamma_normal : Subgroup.Normal (Gamma N) :=
  SLMOD(N).normal_ker


theorem Gamma_one_top : Gamma 1 = ⊤ := by
  /-
    ⊢ Eq (CongruenceSubgroup.Gamma 1) Top.top
  -/
  ext
  /-
    case h
    x✝ : Matrix.SpecialLinearGroup (Fin 2) Int
    ⊢ Iff (Membership.mem (CongruenceSubgroup.Gamma 1) x✝) (Membership.mem Top.top …
  -/
  simp [eq_iff_true_of_subsingleton]
  /-
    🎉 no goals
  -/


lemma mem_Gamma_one (γ : SL(2, ℤ)) : γ ∈ Γ(1) := by
  /-
    γ : Matrix.SpecialLinearGroup (Fin 2) Int
    ⊢ Membership.mem (CongruenceSubgroup.Gamma 1) γ
  -/
  simp only [Gamma_one_top, Subgroup.mem_top]
  /-
    🎉 no goals
  -/


theorem Gamma_zero_bot : Gamma 0 = ⊥ := by
  /-
    ⊢ Eq (CongruenceSubgroup.Gamma 0) Bot.bot
  -/
  ext
  simp only [Gamma_mem, coe_matrix_coe, Int.coe_castRingHom, map_apply, Int.cast_id,
    Subgroup.mem_bot]
  /-
    case h
    x✝ : Matrix.SpecialLinearGroup (Fin 2) Int
    ⊢ Iff (And (Eq (↑(↑x✝ 0 0)) 1) (And (Eq (↑(↑x✝ 0 1)) 0) (And (Eq (↑(↑x✝ 1 0))  …
  -/
  constructor
    /-
      case h.mp
      x✝ : Matrix.SpecialLinearGroup (Fin 2) Int
      ⊢ And (Eq (↑(↑x✝ 0 0)) 1) (And (Eq (↑(↑x✝ 0 1)) 0) (And (Eq (↑(↑x✝ 1 0)) 0) (E …
    -/
  · intro h
    /-
      case h.mp
      x✝ : Matrix.SpecialLinearGroup (Fin 2) Int
      h : And (Eq (↑(↑x✝ 0 0)) 1) (And (Eq (↑(↑x✝ 0 1)) 0) (And (Eq (↑(↑x✝ 1 0)) 0)  …
      ⊢ Eq x✝ 1
    -/
    ext i j
    /-
      case h.mp.a
      x✝ : Matrix.SpecialLinearGroup (Fin 2) Int
      h : And (Eq (↑(↑x✝ 0 0)) 1) (And (Eq (↑(↑x✝ 0 1)) 0) (And (Eq (↑(↑x✝ 1 0)) 0)  …
      i j : Fin 2
      ⊢ Eq (↑x✝ i j) (↑1 i j)
    -/
    fin_cases i <;> fin_cases j <;> simp only [h]
    /-
      case h.mp.a.«0».«0»
      x✝ : Matrix.SpecialLinearGroup (Fin 2) Int
      h : And (Eq (↑(↑x✝ 0 0)) 1) (And (Eq (↑(↑x✝ 0 1)) 0) (And (Eq (↑(↑x✝ 1 0)) 0)  …
      ⊢ Eq (↑x✝ ⟨0, ⋯⟩ ⟨0, ⋯⟩) (↑1 ⟨0, ⋯⟩ ⟨0, ⋯⟩)
    -/
    exacts [h.1, h.2.1, h.2.2.1, h.2.2.2]
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      x✝ : Matrix.SpecialLinearGroup (Fin 2) Int
      ⊢ Eq x✝ 1 → And (Eq (↑(↑x✝ 0 0)) 1) (And (Eq (↑(↑x✝ 0 1)) 0) (And (Eq (↑(↑x✝ 1 …
    -/
  · intro h
    /-
      case h.mpr
      x✝ : Matrix.SpecialLinearGroup (Fin 2) Int
      h : Eq x✝ 1
      ⊢ And (Eq (↑(↑x✝ 0 0)) 1) (And (Eq (↑(↑x✝ 0 1)) 0) (And (Eq (↑(↑x✝ 1 0)) 0) (E …
    -/
    simp [h]
    /-
      🎉 no goals
    -/


lemma ModularGroup_T_pow_mem_Gamma (N M : ℤ) (hNM : N ∣ M) :
    (ModularGroup.T ^ M) ∈ Gamma (Int.natAbs N) := by
  simp only [Gamma_mem, Fin.isValue, ModularGroup.coe_T_zpow, of_apply, cons_val', cons_val_zero,
    empty_val', cons_val_fin_one, Int.cast_one, cons_val_one, head_cons, head_fin_const,
    Int.cast_zero, and_self, and_true, true_and]
  /-
    N M : Int
    hNM : Dvd.dvd N M
    ⊢ Eq (↑M) 0
  -/
  refine Iff.mpr (ZMod.intCast_zmod_eq_zero_iff_dvd M (Int.natAbs N)) ?_
  /-
    N M : Int
    hNM : Dvd.dvd N M
    ⊢ Dvd.dvd (↑N.natAbs) M
  -/
  simp only [Int.natCast_natAbs, abs_dvd, hNM]
  /-
    🎉 no goals
  -/


/-- The congruence subgroup of `SL(2, ℤ)` of matrices whose lower left-hand entry reduces to zero
modulo `N`. -/
def Gamma0 : Subgroup SL(2, ℤ) where
  carrier := { g | (g 1 0 : ZMod N) = 0 }
                 /-
                   N : Nat
                   ⊢ Membership.mem { carrier := setOf fun g => Eq (↑(↑g 1 0)) 0, mul_mem' := ⋯ } …
                 -/
  one_mem' := by simp
    /-
      N : Nat
      ⊢ ∀ {a b : Matrix.SpecialLinearGroup (Fin 2) Int}, Membership.mem (setOf fun g …
    -/
                 /-
                   🎉 no goals
                 -/
    /-
      N : Nat
      a b : Matrix.SpecialLinearGroup (Fin 2) Int
      ha : Membership.mem (setOf fun g => Eq (↑(↑g 1 0)) 0) a
      hb : Membership.mem (setOf fun g => Eq (↑(↑g 1 0)) 0) b
      ⊢ Membership.mem (setOf fun g => Eq (↑(↑g 1 0)) 0) (HMul.hMul a b)
    -/
  mul_mem' := by
    /-
      N : Nat
      a b : Matrix.SpecialLinearGroup (Fin 2) Int
      ha : Membership.mem (setOf fun g => Eq (↑(↑g 1 0)) 0) a
      hb : Membership.mem (setOf fun g => Eq (↑(↑g 1 0)) 0) b
      ⊢ Eq (↑(↑(HMul.hMul a b) 1 0)) 0
    -/
    intro a b ha hb
    /-
      N : Nat
      a b : Matrix.SpecialLinearGroup (Fin 2) Int
      ha : Membership.mem (setOf fun g => Eq (↑(↑g 1 0)) 0) a
      hb : Membership.mem (setOf fun g => Eq (↑(↑g 1 0)) 0) b
      h : Eq (HMul.hMul (↑a) (↑b) 1 0) (HAdd.hAdd (HMul.hMul (↑a 1 0) (↑b 0 0)) (HMu …
      ⊢ Eq (↑(↑(HMul.hMul a b) 1 0)) 0
    -/
    simp only [Set.mem_setOf_eq]
    /-
      N : Nat
      a b : Matrix.SpecialLinearGroup (Fin 2) Int
      ha : Eq (↑(↑a 1 0)) 0
      hb : Eq (↑(↑b 1 0)) 0
      h : Eq (HMul.hMul (↑a) (↑b) 1 0) (HAdd.hAdd (HMul.hMul (↑a 1 0) (↑b 0 0)) (HMu …
      ⊢ Eq (↑(HMul.hMul (↑a) (↑b) 1 0)) 0
    -/
    have h := (Matrix.two_mul_expl a.1 b.1).2.2.1
    /-
      N : Nat
      a b : Matrix.SpecialLinearGroup (Fin 2) Int
      ha : Eq (↑(↑a 1 0)) 0
      hb : Eq (↑(↑b 1 0)) 0
      h : Eq (HMul.hMul (↑a) (↑b) 1 0) (HAdd.hAdd (HMul.hMul (↑a 1 0) (↑b 0 0)) (HMu …
      ⊢ Eq (↑(HAdd.hAdd (HMul.hMul (↑a 1 0) (↑b 0 0)) (HMul.hMul (↑a 1 1) (↑b 1 0))) …
    -/
    simp only [coe_matrix_coe, coe_mul, Int.coe_castRingHom, map_apply, Set.mem_setOf_eq] at *
    /-
      🎉 no goals
    -/
    rw [h]
    simp [ha, hb]
  inv_mem' := by
    /-
      N : Nat
      ⊢ ∀ {x : Matrix.SpecialLinearGroup (Fin 2) Int}, Membership.mem { carrier := s …
    -/
    intro a ha
    /-
      N : Nat
      a : Matrix.SpecialLinearGroup (Fin 2) Int
      ha : Membership.mem { carrier := setOf fun g => Eq (↑(↑g 1 0)) 0, mul_mem' :=  …
      ⊢ Membership.mem { carrier := setOf fun g => Eq (↑(↑g 1 0)) 0, mul_mem' := ⋯,  …
    -/
    simp only [Set.mem_setOf_eq]
    /-
      N : Nat
      a : Matrix.SpecialLinearGroup (Fin 2) Int
      ha : Membership.mem { carrier := setOf fun g => Eq (↑(↑g 1 0)) 0, mul_mem' :=  …
      ⊢ Eq (↑(↑(Inv.inv a) 1 0)) 0
    -/
    rw [SL2_inv_expl a]
    simp only [cons_val_zero, cons_val_one, head_cons, coe_matrix_coe,
      coe_mk, Int.coe_castRingHom, map_apply, Int.cast_neg, neg_eq_zero, Set.mem_setOf_eq] at *
    /-
      N : Nat
      a : Matrix.SpecialLinearGroup (Fin 2) Int
      ha : Eq (↑(↑a 1 0)) 0
      ⊢ Eq (↑(↑a 1 0)) 0
    -/
    exact ha
    /-
      🎉 no goals
    -/


@[simp]
theorem Gamma0_mem {N} {A : SL(2, ℤ)} : A ∈ Gamma0 N ↔ (A 1 0 : ZMod N) = 0 :=
  Iff.rfl


/-- The group homomorphism from `CongruenceSubgroup.Gamma0` to `ZMod N` given by
mapping a matrix to its lower right-hand entry. -/
def Gamma0Map (N : ℕ) : Gamma0 N →* ZMod N where
  toFun g := g.1 1 1
                 /-
                   N✝ N : Nat
                   ⊢ Eq ((fun g => ↑(↑↑g 1 1)) 1) 1
                 -/
  map_one' := by simp
                 /-
                   🎉 no goals
                 -/
  map_mul' := by
    /-
      N✝ N : Nat
      ⊢ ∀ (x y : Subtype fun x => Membership.mem (CongruenceSubgroup.Gamma0 N) x), E …
    -/
    rintro ⟨A, hA⟩ ⟨B, _⟩
    simp only [MulMemClass.mk_mul_mk, Fin.isValue, coe_mul, (two_mul_expl A.1 B).2.2.2,
      Int.cast_add, Int.cast_mul, Gamma0_mem.mp hA, zero_mul, zero_add]


/-- The congruence subgroup `Gamma1` (as a subgroup of `Gamma0`) of matrices whose bottom
row is congruent to `(0, 1)` modulo `N`. -/
def Gamma1' (N : ℕ) : Subgroup (Gamma0 N) :=
  (Gamma0Map N).ker


@[simp]
theorem Gamma1_mem' {N} {γ : Gamma0 N} : γ ∈ Gamma1' N ↔ Gamma0Map N γ = 1 :=
  Iff.rfl


theorem Gamma1_to_Gamma0_mem {N} (A : Gamma0 N) :
    A ∈ Gamma1' N ↔
    ((A.1 0 0 : ℤ) : ZMod N) = 1 ∧ ((A.1 1 1 : ℤ) : ZMod N) = 1
      ∧ ((A.1 1 0 : ℤ) : ZMod N) = 0 := by
  /-
    N : Nat
    A : Subtype fun x => Membership.mem (CongruenceSubgroup.Gamma0 N) x
    ⊢ Iff (Membership.mem (CongruenceSubgroup.Gamma1' N) A) (And (Eq (↑(↑↑A 0 0))  …
  -/
  constructor
    /-
      case mp
      N : Nat
      A : Subtype fun x => Membership.mem (CongruenceSubgroup.Gamma0 N) x
      ⊢ Membership.mem (CongruenceSubgroup.Gamma1' N) A → And (Eq (↑(↑↑A 0 0)) 1) (A …
    -/
  · intro ha
    /-
      case mp
      N : Nat
      A : Subtype fun x => Membership.mem (CongruenceSubgroup.Gamma0 N) x
      ha : Membership.mem (CongruenceSubgroup.Gamma1' N) A
      ⊢ And (Eq (↑(↑↑A 0 0)) 1) (And (Eq (↑(↑↑A 1 1)) 1) (Eq (↑(↑↑A 1 0)) 0))
    -/
    have adet : (A.1.1.det : ZMod N) = 1 := by simp only [A.1.property, Int.cast_one]
    /-
      case mp
      N : Nat
      A : Subtype fun x => Membership.mem (CongruenceSubgroup.Gamma0 N) x
      ha : Membership.mem (CongruenceSubgroup.Gamma1' N) A
      adet : Eq (↑(↑↑A).det) 1
      ⊢ And (Eq (↑(↑↑A 0 0)) 1) (And (Eq (↑(↑↑A 1 1)) 1) (Eq (↑(↑↑A 1 0)) 0))
    -/
    rw [Matrix.det_fin_two] at adet
    simp only [Gamma1_mem', Gamma0Map, MonoidHom.coe_mk, OneHom.coe_mk, Int.cast_sub,
      Int.cast_mul] at *
    simpa only [Gamma1_mem', Gamma0Map, MonoidHom.coe_mk, OneHom.coe_mk, Int.cast_sub,
      Int.cast_mul, ha, Gamma0_mem.mp A.property, and_self_iff, and_true, mul_one, mul_zero,
      sub_zero] using adet
    /-
      case mpr
      N : Nat
      A : Subtype fun x => Membership.mem (CongruenceSubgroup.Gamma0 N) x
      ⊢ And (Eq (↑(↑↑A 0 0)) 1) (And (Eq (↑(↑↑A 1 1)) 1) (Eq (↑(↑↑A 1 0)) 0)) → Memb …
    -/
  · intro ha
    simp only [Gamma1_mem', Gamma0Map, MonoidHom.coe_mk, coe_matrix_coe,
      Int.coe_castRingHom, map_apply]
    /-
      case mpr
      N : Nat
      A : Subtype fun x => Membership.mem (CongruenceSubgroup.Gamma0 N) x
      ha : And (Eq (↑(↑↑A 0 0)) 1) (And (Eq (↑(↑↑A 1 1)) 1) (Eq (↑(↑↑A 1 0)) 0))
      ⊢ Eq ({ toFun := fun g => ↑(↑↑g 1 1), map_one' := ⋯ } A) 1
    -/
    exact ha.2.1
    /-
      🎉 no goals
    -/


/-- The congruence subgroup `Gamma1` of `SL(2, ℤ)` consisting of matrices
whose bottom row is congruent to `(0,1)` modulo `N`. -/
def Gamma1 (N : ℕ) : Subgroup SL(2, ℤ) :=
  Subgroup.map ((Gamma0 N).subtype.comp (Gamma1' N).subtype) ⊤


@[simp]
theorem Gamma1_mem (N : ℕ) (A : SL(2, ℤ)) : A ∈ Gamma1 N ↔
    (A 0 0 : ZMod N) = 1 ∧ (A 1 1 : ZMod N) = 1 ∧ (A 1 0 : ZMod N) = 0 := by
  /-
    N : Nat
    A : Matrix.SpecialLinearGroup (Fin 2) Int
    ⊢ Iff (Membership.mem (CongruenceSubgroup.Gamma1 N) A) (And (Eq (↑(↑A 0 0)) 1) …
  -/
  constructor
    /-
      case mp
      N : Nat
      A : Matrix.SpecialLinearGroup (Fin 2) Int
      ⊢ Membership.mem (CongruenceSubgroup.Gamma1 N) A → And (Eq (↑(↑A 0 0)) 1) (And …
    -/
  · intro ha
    /-
      case mp
      N : Nat
      A : Matrix.SpecialLinearGroup (Fin 2) Int
      ha : Membership.mem (CongruenceSubgroup.Gamma1 N) A
      ⊢ And (Eq (↑(↑A 0 0)) 1) (And (Eq (↑(↑A 1 1)) 1) (Eq (↑(↑A 1 0)) 0))
    -/
    simp_rw [Gamma1, Subgroup.mem_map] at ha
    /-
      case mp
      N : Nat
      A : Matrix.SpecialLinearGroup (Fin 2) Int
      ha : Exists fun x => And (Membership.mem Top.top x) (Eq (((CongruenceSubgroup. …
      ⊢ And (Eq (↑(↑A 0 0)) 1) (And (Eq (↑(↑A 1 1)) 1) (Eq (↑(↑A 1 0)) 0))
    -/
    obtain ⟨⟨x, hx⟩, hxx⟩ := ha
    /-
      case mp.intro.mk
      N : Nat
      A : Matrix.SpecialLinearGroup (Fin 2) Int
      x : Subtype fun x => Membership.mem (CongruenceSubgroup.Gamma0 N) x
      hx : Membership.mem (CongruenceSubgroup.Gamma1' N) x
      hxx : And (Membership.mem Top.top ⟨x, hx⟩) (Eq (((CongruenceSubgroup.Gamma0 N) …
      ⊢ And (Eq (↑(↑A 0 0)) 1) (And (Eq (↑(↑A 1 1)) 1) (Eq (↑(↑A 1 0)) 0))
    -/
    rw [Gamma1_to_Gamma0_mem] at hx
    /-
      case mp.intro.mk
      N : Nat
      A : Matrix.SpecialLinearGroup (Fin 2) Int
      x : Subtype fun x => Membership.mem (CongruenceSubgroup.Gamma0 N) x
      hx✝ : Membership.mem (CongruenceSubgroup.Gamma1' N) x
      hx : And (Eq (↑(↑↑x 0 0)) 1) (And (Eq (↑(↑↑x 1 1)) 1) (Eq (↑(↑↑x 1 0)) 0))
      hxx : And (Membership.mem Top.top ⟨x, hx✝⟩) (Eq (((CongruenceSubgroup.Gamma0 N …
      ⊢ And (Eq (↑(↑A 0 0)) 1) (And (Eq (↑(↑A 1 1)) 1) (Eq (↑(↑A 1 0)) 0))
    -/
    simp only [Subgroup.mem_top, true_and] at hxx
    /-
      case mp.intro.mk
      N : Nat
      A : Matrix.SpecialLinearGroup (Fin 2) Int
      x : Subtype fun x => Membership.mem (CongruenceSubgroup.Gamma0 N) x
      hx✝ : Membership.mem (CongruenceSubgroup.Gamma1' N) x
      hx : And (Eq (↑(↑↑x 0 0)) 1) (And (Eq (↑(↑↑x 1 1)) 1) (Eq (↑(↑↑x 1 0)) 0))
      hxx : Eq (((CongruenceSubgroup.Gamma0 N).subtype.comp (CongruenceSubgroup.Gamm …
      ⊢ And (Eq (↑(↑A 0 0)) 1) (And (Eq (↑(↑A 1 1)) 1) (Eq (↑(↑A 1 0)) 0))
    -/
    rw [← hxx]
    /-
      case mp.intro.mk
      N : Nat
      A : Matrix.SpecialLinearGroup (Fin 2) Int
      x : Subtype fun x => Membership.mem (CongruenceSubgroup.Gamma0 N) x
      hx✝ : Membership.mem (CongruenceSubgroup.Gamma1' N) x
      hx : And (Eq (↑(↑↑x 0 0)) 1) (And (Eq (↑(↑↑x 1 1)) 1) (Eq (↑(↑↑x 1 0)) 0))
      hxx : Eq (((CongruenceSubgroup.Gamma0 N).subtype.comp (CongruenceSubgroup.Gamm …
      ⊢ And (Eq (↑(↑(((CongruenceSubgroup.Gamma0 N).subtype.comp (CongruenceSubgroup …
    -/
    convert hx
    /-
      🎉 no goals
    -/
    /-
      case mpr
      N : Nat
      A : Matrix.SpecialLinearGroup (Fin 2) Int
      ⊢ And (Eq (↑(↑A 0 0)) 1) (And (Eq (↑(↑A 1 1)) 1) (Eq (↑(↑A 1 0)) 0)) → Members …
    -/
  · intro ha
    /-
      case mpr
      N : Nat
      A : Matrix.SpecialLinearGroup (Fin 2) Int
      ha : And (Eq (↑(↑A 0 0)) 1) (And (Eq (↑(↑A 1 1)) 1) (Eq (↑(↑A 1 0)) 0))
      ⊢ Membership.mem (CongruenceSubgroup.Gamma1 N) A
    -/
    simp_rw [Gamma1, Subgroup.mem_map]
    /-
      case mpr
      N : Nat
      A : Matrix.SpecialLinearGroup (Fin 2) Int
      ha : And (Eq (↑(↑A 0 0)) 1) (And (Eq (↑(↑A 1 1)) 1) (Eq (↑(↑A 1 0)) 0))
      ⊢ Exists fun x => And (Membership.mem Top.top x) (Eq (((CongruenceSubgroup.Gam …
    -/
    have hA : A ∈ Gamma0 N := by simp [ha.right.right, Gamma0_mem]
    have HA : (⟨A, hA⟩ : Gamma0 N) ∈ Gamma1' N := by
      simp only [Gamma1_to_Gamma0_mem, Subgroup.coe_mk, coe_matrix_coe,
        Int.coe_castRingHom, map_apply]
      exact ha
    /-
      case mpr
      N : Nat
      A : Matrix.SpecialLinearGroup (Fin 2) Int
      ha : And (Eq (↑(↑A 0 0)) 1) (And (Eq (↑(↑A 1 1)) 1) (Eq (↑(↑A 1 0)) 0))
      hA : Membership.mem (CongruenceSubgroup.Gamma0 N) A
      HA : Membership.mem (CongruenceSubgroup.Gamma1' N) ⟨A, hA⟩
      ⊢ Exists fun x => And (Membership.mem Top.top x) (Eq (((CongruenceSubgroup.Gam …
    -/
    refine ⟨(⟨(⟨A, hA⟩ : Gamma0 N), HA⟩ : (Gamma1' N : Subgroup (Gamma0 N))), ?_⟩
    /-
      case mpr
      N : Nat
      A : Matrix.SpecialLinearGroup (Fin 2) Int
      ha : And (Eq (↑(↑A 0 0)) 1) (And (Eq (↑(↑A 1 1)) 1) (Eq (↑(↑A 1 0)) 0))
      hA : Membership.mem (CongruenceSubgroup.Gamma0 N) A
      HA : Membership.mem (CongruenceSubgroup.Gamma1' N) ⟨A, hA⟩
      ⊢ And (Membership.mem Top.top ⟨⟨A, hA⟩, HA⟩) (Eq (((CongruenceSubgroup.Gamma0  …
    -/
    simp
    /-
      🎉 no goals
    -/


theorem Gamma1_in_Gamma0 (N : ℕ) : Gamma1 N ≤ Gamma0 N := by
  /-
    N : Nat
    ⊢ LE.le (CongruenceSubgroup.Gamma1 N) (CongruenceSubgroup.Gamma0 N)
  -/
  intro x HA
  /-
    N : Nat
    x : Matrix.SpecialLinearGroup (Fin 2) Int
    HA : Membership.mem (CongruenceSubgroup.Gamma1 N) x
    ⊢ Membership.mem (CongruenceSubgroup.Gamma0 N) x
  -/
  simp only [Gamma0_mem, Gamma1_mem, coe_matrix_coe, Int.coe_castRingHom, map_apply] at *
  /-
    N : Nat
    x : Matrix.SpecialLinearGroup (Fin 2) Int
    HA : And (Eq (↑(↑x 0 0)) 1) (And (Eq (↑(↑x 1 1)) 1) (Eq (↑(↑x 1 0)) 0))
    ⊢ Eq (↑(↑x 1 0)) 0
  -/
  exact HA.2.2
  /-
    🎉 no goals
  -/


/-- A congruence subgroup is a subgroup of `SL(2, ℤ)` which contains some `Gamma N` for some
`(N : ℕ+)`. -/
def IsCongruenceSubgroup (Γ : Subgroup SL(2, ℤ)) : Prop :=
  ∃ N : ℕ+, Gamma N ≤ Γ


theorem isCongruenceSubgroup_trans (H K : Subgroup SL(2, ℤ)) (h : H ≤ K)
    (h2 : IsCongruenceSubgroup H) : IsCongruenceSubgroup K := by
  /-
    H K : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
    h : LE.le H K
    h2 : CongruenceSubgroup.IsCongruenceSubgroup H
    ⊢ CongruenceSubgroup.IsCongruenceSubgroup K
  -/
  obtain ⟨N, hN⟩ := h2
  /-
    case intro
    H K : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
    h : LE.le H K
    N : PNat
    hN : LE.le (CongruenceSubgroup.Gamma ↑N) H
    ⊢ CongruenceSubgroup.IsCongruenceSubgroup K
  -/
  exact ⟨N, le_trans hN h⟩
  /-
    🎉 no goals
  -/


theorem Gamma_is_cong_sub (N : ℕ+) : IsCongruenceSubgroup (Gamma N) :=
         /-
           N : PNat
           ⊢ LE.le (CongruenceSubgroup.Gamma ↑N) (CongruenceSubgroup.Gamma ↑N)
         -/
  ⟨N, by simp only [le_refl]⟩
         /-
           🎉 no goals
         -/


theorem Gamma1_is_congruence (N : ℕ+) : IsCongruenceSubgroup (Gamma1 N) := by
  /-
    N : PNat
    ⊢ CongruenceSubgroup.IsCongruenceSubgroup (CongruenceSubgroup.Gamma1 ↑N)
  -/
  refine ⟨N, ?_⟩
  /-
    N : PNat
    ⊢ LE.le (CongruenceSubgroup.Gamma ↑N) (CongruenceSubgroup.Gamma1 ↑N)
  -/
  intro A hA
  /-
    N : PNat
    A : Matrix.SpecialLinearGroup (Fin 2) Int
    hA : Membership.mem (CongruenceSubgroup.Gamma ↑N) A
    ⊢ Membership.mem (CongruenceSubgroup.Gamma1 ↑N) A
  -/
  simp only [Gamma1_mem, Gamma_mem] at *
  /-
    N : PNat
    A : Matrix.SpecialLinearGroup (Fin 2) Int
    hA : And (Eq (↑(↑A 0 0)) 1) (And (Eq (↑(↑A 0 1)) 0) (And (Eq (↑(↑A 1 0)) 0) (E …
    ⊢ And (Eq (↑(↑A 0 0)) 1) (And (Eq (↑(↑A 1 1)) 1) (Eq (↑(↑A 1 0)) 0))
  -/
  simp only [hA, eq_self_iff_true, and_self_iff]
  /-
    🎉 no goals
  -/


theorem Gamma0_is_congruence (N : ℕ+) : IsCongruenceSubgroup (Gamma0 N) :=
  isCongruenceSubgroup_trans _ _ (Gamma1_in_Gamma0 N) (Gamma1_is_congruence N)


theorem Gamma_cong_eq_self (N : ℕ) (g : ConjAct SL(2, ℤ)) : g • Gamma N = Gamma N := by
  /-
    N : Nat
    g : ConjAct (Matrix.SpecialLinearGroup (Fin 2) Int)
    ⊢ Eq (HSMul.hSMul g (CongruenceSubgroup.Gamma N)) (CongruenceSubgroup.Gamma N)
  -/
  apply Subgroup.Normal.conjAct (Gamma_normal N)
  /-
    🎉 no goals
  -/


theorem conj_cong_is_cong (g : ConjAct SL(2, ℤ)) (Γ : Subgroup SL(2, ℤ))
    (h : IsCongruenceSubgroup Γ) : IsCongruenceSubgroup (g • Γ) := by
  /-
    g : ConjAct (Matrix.SpecialLinearGroup (Fin 2) Int)
    Γ : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
    h : CongruenceSubgroup.IsCongruenceSubgroup Γ
    ⊢ CongruenceSubgroup.IsCongruenceSubgroup (HSMul.hSMul g Γ)
  -/
  obtain ⟨N, HN⟩ := h
  /-
    case intro
    g : ConjAct (Matrix.SpecialLinearGroup (Fin 2) Int)
    Γ : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
    N : PNat
    HN : LE.le (CongruenceSubgroup.Gamma ↑N) Γ
    ⊢ CongruenceSubgroup.IsCongruenceSubgroup (HSMul.hSMul g Γ)
  -/
  refine ⟨N, ?_⟩
  /-
    case intro
    g : ConjAct (Matrix.SpecialLinearGroup (Fin 2) Int)
    Γ : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
    N : PNat
    HN : LE.le (CongruenceSubgroup.Gamma ↑N) Γ
    ⊢ LE.le (CongruenceSubgroup.Gamma ↑N) (HSMul.hSMul g Γ)
  -/
  rw [← Gamma_cong_eq_self N g, Subgroup.pointwise_smul_le_pointwise_smul_iff]
  /-
    case intro
    g : ConjAct (Matrix.SpecialLinearGroup (Fin 2) Int)
    Γ : Subgroup (Matrix.SpecialLinearGroup (Fin 2) Int)
    N : PNat
    HN : LE.le (CongruenceSubgroup.Gamma ↑N) Γ
    ⊢ LE.le (CongruenceSubgroup.Gamma ↑N) Γ
  -/
  exact HN
  /-
    🎉 no goals
  -/


