/-- `rootsOfUnity k M` is the subgroup of elements `m : Mˣ` that satisfy `m ^ k = 1`. -/
def rootsOfUnity (k : ℕ) (M : Type*) [CommMonoid M] : Subgroup Mˣ where
  carrier := {ζ | ζ ^ k = 1}
  one_mem' := one_pow _
                     /-
                       M✝ : Type u_1
                       N : Type u_2
                       G : Type u_3
                       R : Type u_4
                       S : Type u_5
                       F : Type u_6
                       inst✝³ : CommMonoid M✝
                       inst✝² : CommMonoid N
                       inst✝¹ : DivisionCommMonoid G
                       k✝ l k : Nat
                       M : Type u_7
                       inst✝ : CommMonoid M
                       a✝ b✝ : Units M
                       x✝¹ : Membership.mem (setOf fun ζ => Eq (HPow.hPow ζ k) 1) a✝
                       x✝ : Membership.mem (setOf fun ζ => Eq (HPow.hPow ζ k) 1) b✝
                       ⊢ Membership.mem (setOf fun ζ => Eq (HPow.hPow ζ k) 1) (HMul.hMul a✝ b✝)
                     -/
  mul_mem' _ _ := by simp_all only [Set.mem_setOf_eq, mul_pow, one_mul]
                     /-
                       🎉 no goals
                     -/
                   /-
                     M✝ : Type u_1
                     N : Type u_2
                     G : Type u_3
                     R : Type u_4
                     S : Type u_5
                     F : Type u_6
                     inst✝³ : CommMonoid M✝
                     inst✝² : CommMonoid N
                     inst✝¹ : DivisionCommMonoid G
                     k✝ l k : Nat
                     M : Type u_7
                     inst✝ : CommMonoid M
                     x✝¹ : Units M
                     x✝ : Membership.mem { carrier := setOf fun ζ => Eq (HPow.hPow ζ k) 1, mul_mem' …
                     ⊢ Membership.mem { carrier := setOf fun ζ => Eq (HPow.hPow ζ k) 1, mul_mem' := …
                   -/
  inv_mem' _ := by simp_all only [Set.mem_setOf_eq, inv_pow, inv_one]
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem mem_rootsOfUnity (k : ℕ) (ζ : Mˣ) : ζ ∈ rootsOfUnity k M ↔ ζ ^ k = 1 :=
  Iff.rfl


/-- A variant of `mem_rootsOfUnity` using `ζ : M`. -/
theorem mem_rootsOfUnity' (k : ℕ) (ζ : Mˣ) : ζ ∈ rootsOfUnity k M ↔ (ζ : M) ^ k = 1 := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : Units M
    ⊢ Iff (Membership.mem (rootsOfUnity k M) ζ) (Eq (HPow.hPow (↑ζ) k) 1)
  -/
  rw [mem_rootsOfUnity]; norm_cast
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem rootsOfUnity_one (M : Type*) [CommMonoid M] : rootsOfUnity 1 M = ⊥ := by
  /-
    M : Type u_7
    inst✝ : CommMonoid M
    ⊢ Eq (rootsOfUnity 1 M) Bot.bot
  -/
  ext1
  /-
    case h
    M : Type u_7
    inst✝ : CommMonoid M
    x✝ : Units M
    ⊢ Iff (Membership.mem (rootsOfUnity 1 M) x✝) (Membership.mem Bot.bot x✝)
  -/
  simp only [mem_rootsOfUnity, pow_one, Subgroup.mem_bot]
  /-
    🎉 no goals
  -/


@[simp]
lemma rootsOfUnity_zero (M : Type*) [CommMonoid M] : rootsOfUnity 0 M = ⊤ := by
  /-
    M : Type u_7
    inst✝ : CommMonoid M
    ⊢ Eq (rootsOfUnity 0 M) Top.top
  -/
  ext1
  /-
    case h
    M : Type u_7
    inst✝ : CommMonoid M
    x✝ : Units M
    ⊢ Iff (Membership.mem (rootsOfUnity 0 M) x✝) (Membership.mem Top.top x✝)
  -/
  simp only [mem_rootsOfUnity, pow_zero, Subgroup.mem_top]
  /-
    🎉 no goals
  -/


theorem rootsOfUnity.coe_injective {n : ℕ} :
    Function.Injective (fun x : rootsOfUnity n M ↦ x.val.val) :=
  Units.ext.comp fun _ _ ↦ Subtype.eq


/-- Make an element of `rootsOfUnity` from a member of the base ring, and a proof that it has
a positive power equal to one. -/
@[simps! coe_val]
def rootsOfUnity.mkOfPowEq (ζ : M) {n : ℕ} [NeZero n] (h : ζ ^ n = 1) : rootsOfUnity n M :=
  ⟨Units.ofPowEqOne ζ n h <| NeZero.ne n, Units.pow_ofPowEqOne _ _⟩


@[simp]
theorem rootsOfUnity.coe_mkOfPowEq {ζ : M} {n : ℕ} [NeZero n] (h : ζ ^ n = 1) :
    ((rootsOfUnity.mkOfPowEq _ h : Mˣ) : M) = ζ :=
  rfl


theorem rootsOfUnity_le_of_dvd (h : k ∣ l) : rootsOfUnity k M ≤ rootsOfUnity l M := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    k l : Nat
    h : Dvd.dvd k l
    ⊢ LE.le (rootsOfUnity k M) (rootsOfUnity l M)
  -/
  obtain ⟨d, rfl⟩ := h
  /-
    case intro
    M : Type u_1
    inst✝ : CommMonoid M
    k d : Nat
    ⊢ LE.le (rootsOfUnity k M) (rootsOfUnity (HMul.hMul k d) M)
  -/
  intro ζ h
  /-
    case intro
    M : Type u_1
    inst✝ : CommMonoid M
    k d : Nat
    ζ : Units M
    h : Membership.mem (rootsOfUnity k M) ζ
    ⊢ Membership.mem (rootsOfUnity (HMul.hMul k d) M) ζ
  -/
  simp_all only [mem_rootsOfUnity, pow_mul, one_pow]
  /-
    🎉 no goals
  -/


theorem map_rootsOfUnity (f : Mˣ →* Nˣ) (k : ℕ) : (rootsOfUnity k M).map f ≤ rootsOfUnity k N := by
  /-
    M : Type u_1
    N : Type u_2
    inst✝¹ : CommMonoid M
    inst✝ : CommMonoid N
    f : MonoidHom (Units M) (Units N)
    k : Nat
    ⊢ LE.le (Subgroup.map f (rootsOfUnity k M)) (rootsOfUnity k N)
  -/
  rintro _ ⟨ζ, h, rfl⟩
  /-
    case intro.intro
    M : Type u_1
    N : Type u_2
    inst✝¹ : CommMonoid M
    inst✝ : CommMonoid N
    f : MonoidHom (Units M) (Units N)
    k : Nat
    ζ : Units M
    h : Membership.mem (↑(rootsOfUnity k M)) ζ
    ⊢ Membership.mem (rootsOfUnity k N) (f ζ)
  -/
  simp_all only [← map_pow, mem_rootsOfUnity, SetLike.mem_coe, MonoidHom.map_one]
  /-
    🎉 no goals
  -/


@[norm_cast]
theorem rootsOfUnity.coe_pow [CommMonoid R] (ζ : rootsOfUnity k R) (m : ℕ) :
    (((ζ ^ m :) : Rˣ) : R) = ((ζ : Rˣ) : R) ^ m := by
  /-
    R : Type u_4
    k : Nat
    inst✝ : CommMonoid R
    ζ : Subtype fun x => Membership.mem (rootsOfUnity k R) x
    m : Nat
    ⊢ Eq (↑↑(HPow.hPow ζ m)) (HPow.hPow (↑↑ζ) m)
  -/
  rw [Subgroup.coe_pow, Units.val_pow_eq_pow_val]
  /-
    🎉 no goals
  -/


/-- The canonical isomorphism from the `n`th roots of unity in `Mˣ`
to the `n`th roots of unity in `M`. -/
def rootsOfUnityUnitsMulEquiv (M : Type*) [CommMonoid M] (n : ℕ) :
    rootsOfUnity n Mˣ ≃* rootsOfUnity n M where
  toFun ζ := ⟨ζ.val, (mem_rootsOfUnity ..).mpr <| (mem_rootsOfUnity' ..).mp ζ.prop⟩
  invFun ζ := ⟨toUnits ζ.val, by
    /-
      M✝ : Type u_1
      N : Type u_2
      G : Type u_3
      R : Type u_4
      S : Type u_5
      F : Type u_6
      inst✝³ : CommMonoid M✝
      inst✝² : CommMonoid N
      inst✝¹ : DivisionCommMonoid G
      k l : Nat
      M : Type u_7
      inst✝ : CommMonoid M
      n : Nat
      ζ : Subtype fun x => Membership.mem (rootsOfUnity n M) x
      ⊢ Membership.mem (rootsOfUnity n (Units M)) (toUnits ↑ζ)
    -/
    simp only [mem_rootsOfUnity, ← map_pow, EmbeddingLike.map_eq_one_iff]
    /-
      M✝ : Type u_1
      N : Type u_2
      G : Type u_3
      R : Type u_4
      S : Type u_5
      F : Type u_6
      inst✝³ : CommMonoid M✝
      inst✝² : CommMonoid N
      inst✝¹ : DivisionCommMonoid G
      k l : Nat
      M : Type u_7
      inst✝ : CommMonoid M
      n : Nat
      ζ : Subtype fun x => Membership.mem (rootsOfUnity n M) x
      ⊢ Eq (HPow.hPow (↑ζ) n) 1
    -/
    exact (mem_rootsOfUnity ..).mp ζ.prop⟩
    /-
      🎉 no goals
    -/
                   /-
                     M✝ : Type u_1
                     N : Type u_2
                     G : Type u_3
                     R : Type u_4
                     S : Type u_5
                     F : Type u_6
                     inst✝³ : CommMonoid M✝
                     inst✝² : CommMonoid N
                     inst✝¹ : DivisionCommMonoid G
                     k l : Nat
                     M : Type u_7
                     inst✝ : CommMonoid M
                     n : Nat
                     ζ : Subtype fun x => Membership.mem (rootsOfUnity n (Units M)) x
                     ⊢ Eq ((fun ζ => ⟨toUnits ↑ζ, ⋯⟩) ((fun ζ => ⟨↑↑ζ, ⋯⟩) ζ)) ζ
                   -/
  left_inv ζ := by simp only [toUnits_val_apply, Subtype.coe_eta]
                   /-
                     🎉 no goals
                   -/
                    /-
                      M✝ : Type u_1
                      N : Type u_2
                      G : Type u_3
                      R : Type u_4
                      S : Type u_5
                      F : Type u_6
                      inst✝³ : CommMonoid M✝
                      inst✝² : CommMonoid N
                      inst✝¹ : DivisionCommMonoid G
                      k l : Nat
                      M : Type u_7
                      inst✝ : CommMonoid M
                      n : Nat
                      ζ : Subtype fun x => Membership.mem (rootsOfUnity n M) x
                      ⊢ Eq ((fun ζ => ⟨↑↑ζ, ⋯⟩) ((fun ζ => ⟨toUnits ↑ζ, ⋯⟩) ζ)) ζ
                    -/
  right_inv ζ := by simp only [val_toUnits_apply, Subtype.coe_eta]
                    /-
                      🎉 no goals
                    -/
                      /-
                        M✝ : Type u_1
                        N : Type u_2
                        G : Type u_3
                        R : Type u_4
                        S : Type u_5
                        F : Type u_6
                        inst✝³ : CommMonoid M✝
                        inst✝² : CommMonoid N
                        inst✝¹ : DivisionCommMonoid G
                        k l : Nat
                        M : Type u_7
                        inst✝ : CommMonoid M
                        n : Nat
                        ζ ζ' : Subtype fun x => Membership.mem (rootsOfUnity n (Units M)) x
                        ⊢ Eq ({ toFun := fun ζ => ⟨↑↑ζ, ⋯⟩, invFun := fun ζ => ⟨toUnits ↑ζ, ⋯⟩, left_i …
                      -/
  map_mul' ζ ζ' := by simp only [Subgroup.coe_mul, Units.val_mul, MulMemClass.mk_mul_mk]
                      /-
                        🎉 no goals
                      -/


/-- Restrict a ring homomorphism to the nth roots of unity. -/
def restrictRootsOfUnity [MonoidHomClass F R S] (σ : F) (n : ℕ) :
    rootsOfUnity n R →* rootsOfUnity n S :=
  { toFun := fun ξ ↦ ⟨Units.map σ (ξ : Rˣ), by
      /-
        M : Type u_1
        N : Type u_2
        G : Type u_3
        R : Type u_4
        S : Type u_5
        F : Type u_6
        inst✝⁶ : CommMonoid M
        inst✝⁵ : CommMonoid N
        inst✝⁴ : DivisionCommMonoid G
        k l : Nat
        inst✝³ : CommMonoid R
        inst✝² : CommMonoid S
        inst✝¹ : FunLike F R S
        inst✝ : MonoidHomClass F R S
        σ : F
        n : Nat
        ξ : Subtype fun x => Membership.mem (rootsOfUnity n R) x
        ⊢ Membership.mem (rootsOfUnity n S) ((Units.map ↑σ) ↑ξ)
      -/
      rw [mem_rootsOfUnity, ← map_pow, Units.ext_iff, Units.coe_map, ξ.prop]
      /-
        M : Type u_1
        N : Type u_2
        G : Type u_3
        R : Type u_4
        S : Type u_5
        F : Type u_6
        inst✝⁶ : CommMonoid M
        inst✝⁵ : CommMonoid N
        inst✝⁴ : DivisionCommMonoid G
        k l : Nat
        inst✝³ : CommMonoid R
        inst✝² : CommMonoid S
        inst✝¹ : FunLike F R S
        inst✝ : MonoidHomClass F R S
        σ : F
        n : Nat
        ξ : Subtype fun x => Membership.mem (rootsOfUnity n R) x
        ⊢ Eq (↑σ ↑1) ↑1
      -/
      exact map_one σ⟩
      /-
        🎉 no goals
      -/
                   /-
                     M : Type u_1
                     N : Type u_2
                     G : Type u_3
                     R : Type u_4
                     S : Type u_5
                     F : Type u_6
                     inst✝⁶ : CommMonoid M
                     inst✝⁵ : CommMonoid N
                     inst✝⁴ : DivisionCommMonoid G
                     k l : Nat
                     inst✝³ : CommMonoid R
                     inst✝² : CommMonoid S
                     inst✝¹ : FunLike F R S
                     inst✝ : MonoidHomClass F R S
                     σ : F
                     n : Nat
                     ⊢ Eq ((fun ξ => ⟨(Units.map ↑σ) ↑ξ, ⋯⟩) 1) 1
                   -/
    map_one' := by ext1; simp only [OneMemClass.coe_one, map_one]
                         /-
                           🎉 no goals
                         -/
    map_mul' := fun ξ₁ ξ₂ ↦ by
      /-
        M : Type u_1
        N : Type u_2
        G : Type u_3
        R : Type u_4
        S : Type u_5
        F : Type u_6
        inst✝⁶ : CommMonoid M
        inst✝⁵ : CommMonoid N
        inst✝⁴ : DivisionCommMonoid G
        k l : Nat
        inst✝³ : CommMonoid R
        inst✝² : CommMonoid S
        inst✝¹ : FunLike F R S
        inst✝ : MonoidHomClass F R S
        σ : F
        n : Nat
        ξ₁ ξ₂ : Subtype fun x => Membership.mem (rootsOfUnity n R) x
        ⊢ Eq ({ toFun := fun ξ => ⟨(Units.map ↑σ) ↑ξ, ⋯⟩, map_one' := ⋯ }.toFun (HMul. …
      -/
      ext1; simp only [Subgroup.coe_mul, map_mul, MulMemClass.mk_mul_mk] }
            /-
              🎉 no goals
            -/


@[simp]
theorem restrictRootsOfUnity_coe_apply [MonoidHomClass F R S] (σ : F) (ζ : rootsOfUnity k R) :
    (restrictRootsOfUnity σ k ζ : Sˣ) = σ (ζ : Rˣ) :=
  rfl


/-- Restrict a monoid isomorphism to the nth roots of unity. -/
nonrec def MulEquiv.restrictRootsOfUnity (σ : R ≃* S) (n : ℕ) :
    rootsOfUnity n R ≃* rootsOfUnity n S where
  toFun := restrictRootsOfUnity σ n
  invFun := restrictRootsOfUnity σ.symm n
                   /-
                     M : Type u_1
                     N : Type u_2
                     G : Type u_3
                     R : Type u_4
                     S : Type u_5
                     F : Type u_6
                     inst✝⁵ : CommMonoid M
                     inst✝⁴ : CommMonoid N
                     inst✝³ : DivisionCommMonoid G
                     k l : Nat
                     inst✝² : CommMonoid R
                     inst✝¹ : CommMonoid S
                     inst✝ : FunLike F R S
                     σ : MulEquiv R S
                     n : Nat
                     ξ : Subtype fun x => Membership.mem (rootsOfUnity n R) x
                     ⊢ Eq ((_root_.restrictRootsOfUnity σ.symm n) ((_root_.restrictRootsOfUnity σ n …
                   -/
  left_inv ξ := by ext; exact σ.symm_apply_apply _
                        /-
                          🎉 no goals
                        -/
                    /-
                      M : Type u_1
                      N : Type u_2
                      G : Type u_3
                      R : Type u_4
                      S : Type u_5
                      F : Type u_6
                      inst✝⁵ : CommMonoid M
                      inst✝⁴ : CommMonoid N
                      inst✝³ : DivisionCommMonoid G
                      k l : Nat
                      inst✝² : CommMonoid R
                      inst✝¹ : CommMonoid S
                      inst✝ : FunLike F R S
                      σ : MulEquiv R S
                      n : Nat
                      ξ : Subtype fun x => Membership.mem (rootsOfUnity n S) x
                      ⊢ Eq ((_root_.restrictRootsOfUnity σ n) ((_root_.restrictRootsOfUnity σ.symm n …
                    -/
  right_inv ξ := by ext; exact σ.apply_symm_apply _
                         /-
                           🎉 no goals
                         -/
  map_mul' := (restrictRootsOfUnity _ n).map_mul


@[simp]
theorem MulEquiv.restrictRootsOfUnity_coe_apply (σ : R ≃* S) (ζ : rootsOfUnity k R) :
    (σ.restrictRootsOfUnity k ζ : Sˣ) = σ (ζ : Rˣ) :=
  rfl


@[simp]
theorem MulEquiv.restrictRootsOfUnity_symm (σ : R ≃* S) :
    (σ.restrictRootsOfUnity k).symm = σ.symm.restrictRootsOfUnity k :=
  rfl


theorem mem_rootsOfUnity_iff_mem_nthRoots {ζ : Rˣ} :
    ζ ∈ rootsOfUnity k R ↔ (ζ : R) ∈ nthRoots k (1 : R) := by
  simp only [mem_rootsOfUnity, mem_nthRoots (NeZero.pos k), Units.ext_iff, Units.val_one,
    Units.val_pow_eq_pow_val]


/-- Equivalence between the `k`-th roots of unity in `R` and the `k`-th roots of `1`.

This is implemented as equivalence of subtypes,
because `rootsOfUnity` is a subgroup of the group of units,
whereas `nthRoots` is a multiset. -/
def rootsOfUnityEquivNthRoots : rootsOfUnity k R ≃ { x // x ∈ nthRoots k (1 : R) } where
  toFun x := ⟨(x : Rˣ), mem_rootsOfUnity_iff_mem_nthRoots.mp x.2⟩
  invFun x := by
    /-
      M : Type u_1
      N : Type u_2
      G : Type u_3
      R : Type u_4
      S : Type u_5
      F : Type u_6
      inst✝⁵ : CommMonoid M
      inst✝⁴ : CommMonoid N
      inst✝³ : DivisionCommMonoid G
      k l : Nat
      inst✝² : NeZero k
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      x : Subtype fun x => Membership.mem (Polynomial.nthRoots k 1) x
      ⊢ Subtype fun x => Membership.mem (rootsOfUnity k R) x
    -/
    refine ⟨⟨x, ↑x ^ (k - 1 : ℕ), ?_, ?_⟩, ?_⟩
    all_goals
      rcases x with ⟨x, hx⟩; rw [mem_nthRoots <| NeZero.pos k] at hx
      simp only [← pow_succ, ← pow_succ', hx, tsub_add_cancel_of_le NeZero.one_le]
    /-
      case refine_3.mk
      M : Type u_1
      N : Type u_2
      G : Type u_3
      R : Type u_4
      S : Type u_5
      F : Type u_6
      inst✝⁵ : CommMonoid M
      inst✝⁴ : CommMonoid N
      inst✝³ : DivisionCommMonoid G
      k l : Nat
      inst✝² : NeZero k
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      x : R
      hx✝ : Membership.mem (Polynomial.nthRoots k 1) x
      hx : Eq (HPow.hPow x k) 1
      ⊢ Membership.mem (rootsOfUnity k R) { val := x, inv := HPow.hPow x (HSub.hSub  …
    -/
    simp only [mem_rootsOfUnity, Units.ext_iff, Units.val_pow_eq_pow_val, hx, Units.val_one]
    /-
      🎉 no goals
    -/
                 /-
                   M : Type u_1
                   N : Type u_2
                   G : Type u_3
                   R : Type u_4
                   S : Type u_5
                   F : Type u_6
                   inst✝⁵ : CommMonoid M
                   inst✝⁴ : CommMonoid N
                   inst✝³ : DivisionCommMonoid G
                   k l : Nat
                   inst✝² : NeZero k
                   inst✝¹ : CommRing R
                   inst✝ : IsDomain R
                   ⊢ Function.LeftInverse (fun x => ⟨{ val := ↑x, inv := HPow.hPow (↑x) (HSub.hSu …
                 -/
  left_inv := by rintro ⟨x, hx⟩; ext; rfl
                                      /-
                                        🎉 no goals
                                      -/
                  /-
                    M : Type u_1
                    N : Type u_2
                    G : Type u_3
                    R : Type u_4
                    S : Type u_5
                    F : Type u_6
                    inst✝⁵ : CommMonoid M
                    inst✝⁴ : CommMonoid N
                    inst✝³ : DivisionCommMonoid G
                    k l : Nat
                    inst✝² : NeZero k
                    inst✝¹ : CommRing R
                    inst✝ : IsDomain R
                    ⊢ Function.RightInverse (fun x => ⟨{ val := ↑x, inv := HPow.hPow (↑x) (HSub.hS …
                  -/
  right_inv := by rintro ⟨x, hx⟩; ext; rfl
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
theorem rootsOfUnityEquivNthRoots_apply (x : rootsOfUnity k R) :
    (rootsOfUnityEquivNthRoots R k x : R) = ((x : Rˣ) : R) :=
  rfl


@[simp]
theorem rootsOfUnityEquivNthRoots_symm_apply (x : { x // x ∈ nthRoots k (1 : R) }) :
    (((rootsOfUnityEquivNthRoots R k).symm x : Rˣ) : R) = (x : R) :=
  rfl


instance rootsOfUnity.fintype : Fintype (rootsOfUnity k R) := by
  classical
  exact Fintype.ofEquiv { x // x ∈ nthRoots k (1 : R) } (rootsOfUnityEquivNthRoots R k).symm


instance rootsOfUnity.isCyclic : IsCyclic (rootsOfUnity k R) :=
  isCyclic_of_subgroup_isDomain ((Units.coeHom R).comp (rootsOfUnity k R).subtype) coe_injective


theorem card_rootsOfUnity : Fintype.card (rootsOfUnity k R) ≤ k := by
  classical
  calc
    Fintype.card (rootsOfUnity k R) = Fintype.card { x // x ∈ nthRoots k (1 : R) } :=
      Fintype.card_congr (rootsOfUnityEquivNthRoots R k)
    _ ≤ Multiset.card (nthRoots k (1 : R)).attach := Multiset.card_le_card (Multiset.dedup_le _)
    _ = Multiset.card (nthRoots k (1 : R)) := Multiset.card_attach
    _ ≤ k := card_nthRoots k 1


theorem map_rootsOfUnity_eq_pow_self [FunLike F R R] [RingHomClass F R R] (σ : F)
    (ζ : rootsOfUnity k R) :
    ∃ m : ℕ, σ (ζ : Rˣ) = ((ζ : Rˣ) : R) ^ m := by
  /-
    R : Type u_4
    F : Type u_6
    k : Nat
    inst✝⁴ : NeZero k
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : FunLike F R R
    inst✝ : RingHomClass F R R
    σ : F
    ζ : Subtype fun x => Membership.mem (rootsOfUnity k R) x
    ⊢ Exists fun m => Eq (σ ↑↑ζ) (HPow.hPow (↑↑ζ) m)
  -/
  obtain ⟨m, hm⟩ := MonoidHom.map_cyclic (restrictRootsOfUnity σ k)
  rw [← restrictRootsOfUnity_coe_apply, hm, ← zpow_mod_orderOf, ← Int.toNat_of_nonneg
      (m.emod_nonneg (Int.natCast_ne_zero.mpr (pos_iff_ne_zero.mp (orderOf_pos ζ)))),
    zpow_natCast, rootsOfUnity.coe_pow]
  /-
    case intro
    R : Type u_4
    F : Type u_6
    k : Nat
    inst✝⁴ : NeZero k
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : FunLike F R R
    inst✝ : RingHomClass F R R
    σ : F
    ζ : Subtype fun x => Membership.mem (rootsOfUnity k R) x
    m : Int
    hm : ∀ (g : Subtype fun x => Membership.mem (rootsOfUnity k R) x), Eq ((restri …
    ⊢ Exists fun m_1 => Eq (HPow.hPow (↑↑ζ) (HMod.hMod m ↑(orderOf ζ)).toNat) (HPo …
  -/
  exact ⟨(m % orderOf ζ).toNat, rfl⟩
  /-
    🎉 no goals
  -/


theorem mem_rootsOfUnity_prime_pow_mul_iff (p k : ℕ) (m : ℕ) [ExpChar R p] {ζ : Rˣ} :
    ζ ∈ rootsOfUnity (p ^ k * m) R ↔ ζ ∈ rootsOfUnity m R := by
  /-
    R : Type u_4
    inst✝² : CommRing R
    inst✝¹ : IsReduced R
    p k m : Nat
    inst✝ : ExpChar R p
    ζ : Units R
    ⊢ Iff (Membership.mem (rootsOfUnity (HMul.hMul (HPow.hPow p k) m) R) ζ) (Membe …
  -/
  simp only [mem_rootsOfUnity', ExpChar.pow_prime_pow_mul_eq_one_iff]
  /-
    🎉 no goals
  -/


/-- A variant of `mem_rootsOfUnity_prime_pow_mul_iff` in terms of `ζ ^ _`.-/
@[simp]
theorem mem_rootsOfUnity_prime_pow_mul_iff' (p k : ℕ) (m : ℕ) [ExpChar R p] {ζ : Rˣ} :
    ζ ^ (p ^ k * m) = 1 ↔ ζ ∈ rootsOfUnity m R := by
  /-
    R : Type u_4
    inst✝² : CommRing R
    inst✝¹ : IsReduced R
    p k m : Nat
    inst✝ : ExpChar R p
    ζ : Units R
    ⊢ Iff (Eq (HPow.hPow ζ (HMul.hMul (HPow.hPow p k) m)) 1) (Membership.mem (root …
  -/
  rw [← mem_rootsOfUnity, mem_rootsOfUnity_prime_pow_mul_iff]
  /-
    🎉 no goals
  -/


/-- The isomorphism from the group of group homomorphisms from a finite cyclic group `G` of order
`n` into another group `G'` to the group of `n`th roots of unity in `G'` determined by a generator
`g` of `G`. It sends `φ : G →* G'` to `φ g`. -/
noncomputable
def monoidHomMulEquivRootsOfUnityOfGenerator {G : Type*} [CommGroup G] {g : G}
    (hg : ∀ (x : G), x ∈ Subgroup.zpowers g) (G' : Type*) [CommGroup G'] :
    (G →* G') ≃* rootsOfUnity (Nat.card G) G' where
  toFun φ := ⟨(IsUnit.map φ <| Group.isUnit g).unit, by
    simp only [mem_rootsOfUnity, Units.ext_iff, Units.val_pow_eq_pow_val, IsUnit.unit_spec,
      ← map_pow, pow_card_eq_one', map_one, Units.val_one]⟩
  invFun ζ := monoidHomOfForallMemZpowers hg (g' := (ζ.val : G')) <| by
    simpa only [orderOf_eq_card_of_forall_mem_zpowers hg, orderOf_dvd_iff_pow_eq_one,
      ← Units.val_pow_eq_pow_val, Units.val_eq_one] using ζ.prop
  left_inv φ := (MonoidHom.eq_iff_eq_on_generator hg _ φ).mpr <| by
    /-
      M : Type u_1
      N : Type u_2
      G✝ : Type u_3
      R : Type u_4
      S : Type u_5
      F : Type u_6
      inst✝⁴ : CommMonoid M
      inst✝³ : CommMonoid N
      inst✝² : DivisionCommMonoid G✝
      G : Type u_7
      inst✝¹ : CommGroup G
      g : G
      hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
      G' : Type u_8
      inst✝ : CommGroup G'
      φ : MonoidHom G G'
      ⊢ Eq (((fun ζ => monoidHomOfForallMemZpowers hg ⋯) ((fun φ => ⟨⋯.unit, ⋯⟩) φ)) …
    -/
    simp only [IsUnit.unit_spec, monoidHomOfForallMemZpowers_apply_gen]
    /-
      🎉 no goals
    -/
  right_inv φ := Subtype.ext <| by
    /-
      M : Type u_1
      N : Type u_2
      G✝ : Type u_3
      R : Type u_4
      S : Type u_5
      F : Type u_6
      inst✝⁴ : CommMonoid M
      inst✝³ : CommMonoid N
      inst✝² : DivisionCommMonoid G✝
      G : Type u_7
      inst✝¹ : CommGroup G
      g : G
      hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
      G' : Type u_8
      inst✝ : CommGroup G'
      φ : Subtype fun x => Membership.mem (rootsOfUnity (Nat.card G) G') x
      ⊢ Eq ↑((fun φ => ⟨⋯.unit, ⋯⟩) ((fun ζ => monoidHomOfForallMemZpowers hg ⋯) φ)) …
    -/
    simp only [monoidHomOfForallMemZpowers_apply_gen, IsUnit.unit_of_val_units]
    /-
      🎉 no goals
    -/
  map_mul' x y := by
    simp only [MonoidHom.mul_apply, MulMemClass.mk_mul_mk, Subtype.mk.injEq, Units.ext_iff,
      IsUnit.unit_spec, Units.val_mul]


/-- The group of group homomorphisms from a finite cyclic group `G` of order `n` into another
group `G'` is (noncanonically) isomorphic to the group of `n`th roots of unity in `G'`. -/
lemma monoidHom_mulEquiv_rootsOfUnity (G : Type*) [CommGroup G] [IsCyclic G]
    (G' : Type*) [CommGroup G'] :
    Nonempty <| (G →* G') ≃* rootsOfUnity (Nat.card G) G' := by
  /-
    G : Type u_7
    inst✝² : CommGroup G
    inst✝¹ : IsCyclic G
    G' : Type u_8
    inst✝ : CommGroup G'
    ⊢ Nonempty (MulEquiv (MonoidHom G G') (Subtype fun x => Membership.mem (rootsO …
  -/
  obtain ⟨g, hg⟩ := IsCyclic.exists_generator (α := G)
  /-
    case intro
    G : Type u_7
    inst✝² : CommGroup G
    inst✝¹ : IsCyclic G
    G' : Type u_8
    inst✝ : CommGroup G'
    g : G
    hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
    ⊢ Nonempty (MulEquiv (MonoidHom G G') (Subtype fun x => Membership.mem (rootsO …
  -/
  exact ⟨monoidHomMulEquivRootsOfUnityOfGenerator hg G'⟩
  /-
    🎉 no goals
  -/


