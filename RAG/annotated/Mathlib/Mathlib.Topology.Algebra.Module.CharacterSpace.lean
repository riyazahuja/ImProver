/-- The character space of a topological algebra is the subset of elements of the weak dual that
are also algebra homomorphisms. -/
def characterSpace (𝕜 : Type*) (A : Type*) [CommSemiring 𝕜] [TopologicalSpace 𝕜] [ContinuousAdd 𝕜]
    [ContinuousConstSMul 𝕜 𝕜] [NonUnitalNonAssocSemiring A] [TopologicalSpace A] [Module 𝕜 A] :=
  {φ : WeakDual 𝕜 A | φ ≠ 0 ∧ ∀ x y : A, φ (x * y) = φ x * φ y}


instance instFunLike : FunLike (characterSpace 𝕜 A) A 𝕜 where
  coe φ := ((φ : WeakDual 𝕜 A) : A → 𝕜)
                             /-
                               𝕜 : Type u_1
                               A : Type u_2
                               inst✝⁶ : CommSemiring 𝕜
                               inst✝⁵ : TopologicalSpace 𝕜
                               inst✝⁴ : ContinuousAdd 𝕜
                               inst✝³ : ContinuousConstSMul 𝕜 𝕜
                               inst✝² : NonUnitalNonAssocSemiring A
                               inst✝¹ : TopologicalSpace A
                               inst✝ : Module 𝕜 A
                               φ ψ : ↑(WeakDual.characterSpace 𝕜 A)
                               h : Eq ((fun φ => ⇑↑φ) φ) ((fun φ => ⇑↑φ) ψ)
                               ⊢ Eq φ ψ
                             -/
  coe_injective' φ ψ h := by ext1; apply DFunLike.ext; exact congr_fun h
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- Elements of the character space are continuous linear maps. -/
instance instContinuousLinearMapClass : ContinuousLinearMapClass (characterSpace 𝕜 A) 𝕜 A 𝕜 where
  map_smulₛₗ φ := (φ : WeakDual 𝕜 A).map_smul
  map_add φ := (φ : WeakDual 𝕜 A).map_add
  map_continuous φ := (φ : WeakDual 𝕜 A).cont

-- Porting note: moved because Lean 4 doesn't see the `DFunLike` instance on `characterSpace 𝕜 A`
-- until the `ContinuousLinearMapClass` instance is declared

@[simp, norm_cast]
protected theorem coe_coe (φ : characterSpace 𝕜 A) : ⇑(φ : WeakDual 𝕜 A) = (φ : A → 𝕜) :=
  rfl


@[ext]
theorem ext {φ ψ : characterSpace 𝕜 A} (h : ∀ x, φ x = ψ x) : φ = ψ :=
  DFunLike.ext _ _ h


/-- An element of the character space, as a continuous linear map. -/
def toCLM (φ : characterSpace 𝕜 A) : A →L[𝕜] 𝕜 :=
  (φ : WeakDual 𝕜 A)


@[simp]
theorem coe_toCLM (φ : characterSpace 𝕜 A) : ⇑(toCLM φ) = φ :=
  rfl


/-- Elements of the character space are non-unital algebra homomorphisms. -/
instance instNonUnitalAlgHomClass : NonUnitalAlgHomClass (characterSpace 𝕜 A) 𝕜 A 𝕜 :=
  { CharacterSpace.instContinuousLinearMapClass with
    map_smulₛₗ := fun φ => map_smul φ
    map_zero := fun φ => map_zero φ
    map_mul := fun φ => φ.prop.2 }


/-- An element of the character space, as a non-unital algebra homomorphism. -/
def toNonUnitalAlgHom (φ : characterSpace 𝕜 A) : A →ₙₐ[𝕜] 𝕜 where
  toFun := (φ : A → 𝕜)
  map_mul' := map_mul φ
  map_smul' := map_smul φ
  map_zero' := map_zero φ
  map_add' := map_add φ


@[simp]
theorem coe_toNonUnitalAlgHom (φ : characterSpace 𝕜 A) : ⇑(toNonUnitalAlgHom φ) = φ :=
  rfl


instance instIsEmpty [Subsingleton A] : IsEmpty (characterSpace 𝕜 A) :=
  ⟨fun φ => φ.prop.1 <|
    ContinuousLinearMap.ext fun x => by
      /-
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁷ : CommSemiring 𝕜
        inst✝⁶ : TopologicalSpace 𝕜
        inst✝⁵ : ContinuousAdd 𝕜
        inst✝⁴ : ContinuousConstSMul 𝕜 𝕜
        inst✝³ : NonUnitalNonAssocSemiring A
        inst✝² : TopologicalSpace A
        inst✝¹ : Module 𝕜 A
        inst✝ : Subsingleton A
        φ : ↑(WeakDual.characterSpace 𝕜 A)
        x : A
        ⊢ Eq (↑φ x) (0 x)
      -/
      rw [show x = 0 from Subsingleton.elim x 0, map_zero, map_zero] ⟩
      /-
        🎉 no goals
      -/


theorem union_zero :
    characterSpace 𝕜 A ∪ {0} = {φ : WeakDual 𝕜 A | ∀ x y : A, φ (x * y) = φ x * φ y} :=
  le_antisymm (by
      /-
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁶ : CommSemiring 𝕜
        inst✝⁵ : TopologicalSpace 𝕜
        inst✝⁴ : ContinuousAdd 𝕜
        inst✝³ : ContinuousConstSMul 𝕜 𝕜
        inst✝² : NonUnitalNonAssocSemiring A
        inst✝¹ : TopologicalSpace A
        inst✝ : Module 𝕜 A
        ⊢ LE.le (Union.union (WeakDual.characterSpace 𝕜 A) (Singleton.singleton 0)) (s …
      -/
      rintro φ (hφ | rfl)
        /-
          case inl
          𝕜 : Type u_1
          A : Type u_2
          inst✝⁶ : CommSemiring 𝕜
          inst✝⁵ : TopologicalSpace 𝕜
          inst✝⁴ : ContinuousAdd 𝕜
          inst✝³ : ContinuousConstSMul 𝕜 𝕜
          inst✝² : NonUnitalNonAssocSemiring A
          inst✝¹ : TopologicalSpace A
          inst✝ : Module 𝕜 A
          φ : WeakDual 𝕜 A
          hφ : Membership.mem (WeakDual.characterSpace 𝕜 A) φ
          ⊢ Membership.mem (setOf fun φ => ∀ (x y : A), Eq (φ (HMul.hMul x y)) (HMul.hMu …
        -/
      · exact hφ.2
        /-
          🎉 no goals
        -/
        /-
          case inr
          𝕜 : Type u_1
          A : Type u_2
          inst✝⁶ : CommSemiring 𝕜
          inst✝⁵ : TopologicalSpace 𝕜
          inst✝⁴ : ContinuousAdd 𝕜
          inst✝³ : ContinuousConstSMul 𝕜 𝕜
          inst✝² : NonUnitalNonAssocSemiring A
          inst✝¹ : TopologicalSpace A
          inst✝ : Module 𝕜 A
          ⊢ Membership.mem (setOf fun φ => ∀ (x y : A), Eq (φ (HMul.hMul x y)) (HMul.hMu …
        -/
      · exact fun _ _ => by exact (zero_mul (0 : 𝕜)).symm)
        /-
          🎉 no goals
        -/
    fun φ hφ => Or.elim (em <| φ = 0) Or.inr fun h₀ => Or.inl ⟨h₀, hφ⟩


/-- The `characterSpace 𝕜 A` along with `0` is always a closed set in `WeakDual 𝕜 A`. -/
theorem union_zero_isClosed [T2Space 𝕜] [ContinuousMul 𝕜] :
    IsClosed (characterSpace 𝕜 A ∪ {0}) := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁸ : CommSemiring 𝕜
    inst✝⁷ : TopologicalSpace 𝕜
    inst✝⁶ : ContinuousAdd 𝕜
    inst✝⁵ : ContinuousConstSMul 𝕜 𝕜
    inst✝⁴ : NonUnitalNonAssocSemiring A
    inst✝³ : TopologicalSpace A
    inst✝² : Module 𝕜 A
    inst✝¹ : T2Space 𝕜
    inst✝ : ContinuousMul 𝕜
    ⊢ IsClosed (Union.union (WeakDual.characterSpace 𝕜 A) (Singleton.singleton 0))
  -/
  simp only [union_zero, Set.setOf_forall]
  exact
    isClosed_iInter fun x =>
      isClosed_iInter fun y =>
        isClosed_eq (eval_continuous _) <| (eval_continuous _).mul (eval_continuous _)


/-- In a unital algebra, elements of the character space are algebra homomorphisms. -/
instance instAlgHomClass : AlgHomClass (characterSpace 𝕜 A) 𝕜 A 𝕜 :=
  haveI map_one' : ∀ φ : characterSpace 𝕜 A, φ 1 = 1 := fun φ => by
    /-
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁷ : CommRing 𝕜
      inst✝⁶ : NoZeroDivisors 𝕜
      inst✝⁵ : TopologicalSpace 𝕜
      inst✝⁴ : ContinuousAdd 𝕜
      inst✝³ : ContinuousConstSMul 𝕜 𝕜
      inst✝² : TopologicalSpace A
      inst✝¹ : Semiring A
      inst✝ : Algebra 𝕜 A
      φ : ↑(WeakDual.characterSpace 𝕜 A)
      ⊢ Eq (φ 1) 1
    -/
    have h₁ : φ 1 * (1 - φ 1) = 0 := by rw [mul_sub, sub_eq_zero, mul_one, ← map_mul φ, one_mul]
    /-
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁷ : CommRing 𝕜
      inst✝⁶ : NoZeroDivisors 𝕜
      inst✝⁵ : TopologicalSpace 𝕜
      inst✝⁴ : ContinuousAdd 𝕜
      inst✝³ : ContinuousConstSMul 𝕜 𝕜
      inst✝² : TopologicalSpace A
      inst✝¹ : Semiring A
      inst✝ : Algebra 𝕜 A
      φ : ↑(WeakDual.characterSpace 𝕜 A)
      h₁ : Eq (HMul.hMul (φ 1) (HSub.hSub 1 (φ 1))) 0
      ⊢ Eq (φ 1) 1
    -/
    rcases mul_eq_zero.mp h₁ with (h₂ | h₂)
      /-
        case inl
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁷ : CommRing 𝕜
        inst✝⁶ : NoZeroDivisors 𝕜
        inst✝⁵ : TopologicalSpace 𝕜
        inst✝⁴ : ContinuousAdd 𝕜
        inst✝³ : ContinuousConstSMul 𝕜 𝕜
        inst✝² : TopologicalSpace A
        inst✝¹ : Semiring A
        inst✝ : Algebra 𝕜 A
        φ : ↑(WeakDual.characterSpace 𝕜 A)
        h₁ : Eq (HMul.hMul (φ 1) (HSub.hSub 1 (φ 1))) 0
        h₂ : Eq (φ 1) 0
        ⊢ Eq (φ 1) 1
      -/
    · have : ∀ a, φ (a * 1) = 0 := fun a => by simp only [map_mul φ, h₂, mul_zero]
      /-
        case inl
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁷ : CommRing 𝕜
        inst✝⁶ : NoZeroDivisors 𝕜
        inst✝⁵ : TopologicalSpace 𝕜
        inst✝⁴ : ContinuousAdd 𝕜
        inst✝³ : ContinuousConstSMul 𝕜 𝕜
        inst✝² : TopologicalSpace A
        inst✝¹ : Semiring A
        inst✝ : Algebra 𝕜 A
        φ : ↑(WeakDual.characterSpace 𝕜 A)
        h₁ : Eq (HMul.hMul (φ 1) (HSub.hSub 1 (φ 1))) 0
        h₂ : Eq (φ 1) 0
        this : ∀ (a : A), Eq (φ (HMul.hMul a 1)) 0
        ⊢ Eq (φ 1) 1
      -/
      exact False.elim (φ.prop.1 <| ContinuousLinearMap.ext <| by simpa only [mul_one] using this)
      /-
        🎉 no goals
      -/
      /-
        case inr
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁷ : CommRing 𝕜
        inst✝⁶ : NoZeroDivisors 𝕜
        inst✝⁵ : TopologicalSpace 𝕜
        inst✝⁴ : ContinuousAdd 𝕜
        inst✝³ : ContinuousConstSMul 𝕜 𝕜
        inst✝² : TopologicalSpace A
        inst✝¹ : Semiring A
        inst✝ : Algebra 𝕜 A
        φ : ↑(WeakDual.characterSpace 𝕜 A)
        h₁ : Eq (HMul.hMul (φ 1) (HSub.hSub 1 (φ 1))) 0
        h₂ : Eq (HSub.hSub 1 (φ 1)) 0
        ⊢ Eq (φ 1) 1
      -/
    · exact (sub_eq_zero.mp h₂).symm
      /-
        🎉 no goals
      -/
  { CharacterSpace.instNonUnitalAlgHomClass with
    map_one := map_one'
    commutes := fun φ r => by
      /-
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁷ : CommRing 𝕜
        inst✝⁶ : NoZeroDivisors 𝕜
        inst✝⁵ : TopologicalSpace 𝕜
        inst✝⁴ : ContinuousAdd 𝕜
        inst✝³ : ContinuousConstSMul 𝕜 𝕜
        inst✝² : TopologicalSpace A
        inst✝¹ : Semiring A
        inst✝ : Algebra 𝕜 A
        map_one' : ∀ (φ : ↑(WeakDual.characterSpace 𝕜 A)), Eq (φ 1) 1
        φ : ↑(WeakDual.characterSpace 𝕜 A)
        r : 𝕜
        ⊢ Eq (φ ((algebraMap 𝕜 A) r)) ((algebraMap 𝕜 𝕜) r)
      -/
      rw [Algebra.algebraMap_eq_smul_one, Algebra.id.map_eq_id, RingHom.id_apply]
      /-
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁷ : CommRing 𝕜
        inst✝⁶ : NoZeroDivisors 𝕜
        inst✝⁵ : TopologicalSpace 𝕜
        inst✝⁴ : ContinuousAdd 𝕜
        inst✝³ : ContinuousConstSMul 𝕜 𝕜
        inst✝² : TopologicalSpace A
        inst✝¹ : Semiring A
        inst✝ : Algebra 𝕜 A
        map_one' : ∀ (φ : ↑(WeakDual.characterSpace 𝕜 A)), Eq (φ 1) 1
        φ : ↑(WeakDual.characterSpace 𝕜 A)
        r : 𝕜
        ⊢ Eq (φ (HSMul.hSMul r 1)) r
      -/
      rw [map_smul, Algebra.id.smul_eq_mul, map_one' φ, mul_one] }
      /-
        🎉 no goals
      -/


/-- An element of the character space of a unital algebra, as an algebra homomorphism. -/
@[simps]
def toAlgHom (φ : characterSpace 𝕜 A) : A →ₐ[𝕜] 𝕜 :=
  { toNonUnitalAlgHom φ with
    map_one' := map_one φ
    commutes' := AlgHomClass.commutes φ }


theorem eq_set_map_one_map_mul [Nontrivial 𝕜] :
    characterSpace 𝕜 A = {φ : WeakDual 𝕜 A | φ 1 = 1 ∧ ∀ x y : A, φ (x * y) = φ x * φ y} := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁸ : CommRing 𝕜
    inst✝⁷ : NoZeroDivisors 𝕜
    inst✝⁶ : TopologicalSpace 𝕜
    inst✝⁵ : ContinuousAdd 𝕜
    inst✝⁴ : ContinuousConstSMul 𝕜 𝕜
    inst✝³ : TopologicalSpace A
    inst✝² : Semiring A
    inst✝¹ : Algebra 𝕜 A
    inst✝ : Nontrivial 𝕜
    ⊢ Eq (WeakDual.characterSpace 𝕜 A) (setOf fun φ => And (Eq (φ 1) 1) (∀ (x y :  …
  -/
  ext φ
  /-
    case h
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁸ : CommRing 𝕜
    inst✝⁷ : NoZeroDivisors 𝕜
    inst✝⁶ : TopologicalSpace 𝕜
    inst✝⁵ : ContinuousAdd 𝕜
    inst✝⁴ : ContinuousConstSMul 𝕜 𝕜
    inst✝³ : TopologicalSpace A
    inst✝² : Semiring A
    inst✝¹ : Algebra 𝕜 A
    inst✝ : Nontrivial 𝕜
    φ : WeakDual 𝕜 A
    ⊢ Iff (Membership.mem (WeakDual.characterSpace 𝕜 A) φ) (Membership.mem (setOf  …
  -/
  refine ⟨?_, ?_⟩
    /-
      case h.refine_1
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁸ : CommRing 𝕜
      inst✝⁷ : NoZeroDivisors 𝕜
      inst✝⁶ : TopologicalSpace 𝕜
      inst✝⁵ : ContinuousAdd 𝕜
      inst✝⁴ : ContinuousConstSMul 𝕜 𝕜
      inst✝³ : TopologicalSpace A
      inst✝² : Semiring A
      inst✝¹ : Algebra 𝕜 A
      inst✝ : Nontrivial 𝕜
      φ : WeakDual 𝕜 A
      ⊢ Membership.mem (WeakDual.characterSpace 𝕜 A) φ → Membership.mem (setOf fun φ …
    -/
  · rintro hφ
    /-
      case h.refine_1
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁸ : CommRing 𝕜
      inst✝⁷ : NoZeroDivisors 𝕜
      inst✝⁶ : TopologicalSpace 𝕜
      inst✝⁵ : ContinuousAdd 𝕜
      inst✝⁴ : ContinuousConstSMul 𝕜 𝕜
      inst✝³ : TopologicalSpace A
      inst✝² : Semiring A
      inst✝¹ : Algebra 𝕜 A
      inst✝ : Nontrivial 𝕜
      φ : WeakDual 𝕜 A
      hφ : Membership.mem (WeakDual.characterSpace 𝕜 A) φ
      ⊢ Membership.mem (setOf fun φ => And (Eq (φ 1) 1) (∀ (x y : A), Eq (φ (HMul.hM …
    -/
    lift φ to characterSpace 𝕜 A using hφ
    /-
      case h.refine_1.intro
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁸ : CommRing 𝕜
      inst✝⁷ : NoZeroDivisors 𝕜
      inst✝⁶ : TopologicalSpace 𝕜
      inst✝⁵ : ContinuousAdd 𝕜
      inst✝⁴ : ContinuousConstSMul 𝕜 𝕜
      inst✝³ : TopologicalSpace A
      inst✝² : Semiring A
      inst✝¹ : Algebra 𝕜 A
      inst✝ : Nontrivial 𝕜
      φ : Subtype fun x => Membership.mem (WeakDual.characterSpace 𝕜 A) x
      ⊢ Membership.mem (setOf fun φ => And (Eq (φ 1) 1) (∀ (x y : A), Eq (φ (HMul.hM …
    -/
    exact ⟨map_one φ, map_mul φ⟩
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁸ : CommRing 𝕜
      inst✝⁷ : NoZeroDivisors 𝕜
      inst✝⁶ : TopologicalSpace 𝕜
      inst✝⁵ : ContinuousAdd 𝕜
      inst✝⁴ : ContinuousConstSMul 𝕜 𝕜
      inst✝³ : TopologicalSpace A
      inst✝² : Semiring A
      inst✝¹ : Algebra 𝕜 A
      inst✝ : Nontrivial 𝕜
      φ : WeakDual 𝕜 A
      ⊢ Membership.mem (setOf fun φ => And (Eq (φ 1) 1) (∀ (x y : A), Eq (φ (HMul.hM …
    -/
  · rintro ⟨hφ₁, hφ₂⟩
    /-
      case h.refine_2.intro
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁸ : CommRing 𝕜
      inst✝⁷ : NoZeroDivisors 𝕜
      inst✝⁶ : TopologicalSpace 𝕜
      inst✝⁵ : ContinuousAdd 𝕜
      inst✝⁴ : ContinuousConstSMul 𝕜 𝕜
      inst✝³ : TopologicalSpace A
      inst✝² : Semiring A
      inst✝¹ : Algebra 𝕜 A
      inst✝ : Nontrivial 𝕜
      φ : WeakDual 𝕜 A
      hφ₁ : Eq (φ 1) 1
      hφ₂ : ∀ (x y : A), Eq (φ (HMul.hMul x y)) (HMul.hMul (φ x) (φ y))
      ⊢ Membership.mem (WeakDual.characterSpace 𝕜 A) φ
    -/
    refine ⟨?_, hφ₂⟩
    /-
      case h.refine_2.intro
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁸ : CommRing 𝕜
      inst✝⁷ : NoZeroDivisors 𝕜
      inst✝⁶ : TopologicalSpace 𝕜
      inst✝⁵ : ContinuousAdd 𝕜
      inst✝⁴ : ContinuousConstSMul 𝕜 𝕜
      inst✝³ : TopologicalSpace A
      inst✝² : Semiring A
      inst✝¹ : Algebra 𝕜 A
      inst✝ : Nontrivial 𝕜
      φ : WeakDual 𝕜 A
      hφ₁ : Eq (φ 1) 1
      hφ₂ : ∀ (x y : A), Eq (φ (HMul.hMul x y)) (HMul.hMul (φ x) (φ y))
      ⊢ Ne φ 0
    -/
    rintro rfl
    /-
      case h.refine_2.intro
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁸ : CommRing 𝕜
      inst✝⁷ : NoZeroDivisors 𝕜
      inst✝⁶ : TopologicalSpace 𝕜
      inst✝⁵ : ContinuousAdd 𝕜
      inst✝⁴ : ContinuousConstSMul 𝕜 𝕜
      inst✝³ : TopologicalSpace A
      inst✝² : Semiring A
      inst✝¹ : Algebra 𝕜 A
      inst✝ : Nontrivial 𝕜
      hφ₁ : Eq (0 1) 1
      hφ₂ : ∀ (x y : A), Eq (0 (HMul.hMul x y)) (HMul.hMul (0 x) (0 y))
      ⊢ False
    -/
    exact zero_ne_one hφ₁
    /-
      🎉 no goals
    -/


/-- under suitable mild assumptions on `𝕜`, the character space is a closed set in
`WeakDual 𝕜 A`. -/
protected theorem isClosed [Nontrivial 𝕜] [T2Space 𝕜] [ContinuousMul 𝕜] :
    IsClosed (characterSpace 𝕜 A) := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝¹⁰ : CommRing 𝕜
    inst✝⁹ : NoZeroDivisors 𝕜
    inst✝⁸ : TopologicalSpace 𝕜
    inst✝⁷ : ContinuousAdd 𝕜
    inst✝⁶ : ContinuousConstSMul 𝕜 𝕜
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Semiring A
    inst✝³ : Algebra 𝕜 A
    inst✝² : Nontrivial 𝕜
    inst✝¹ : T2Space 𝕜
    inst✝ : ContinuousMul 𝕜
    ⊢ IsClosed (WeakDual.characterSpace 𝕜 A)
  -/
  rw [eq_set_map_one_map_mul, Set.setOf_and]
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝¹⁰ : CommRing 𝕜
    inst✝⁹ : NoZeroDivisors 𝕜
    inst✝⁸ : TopologicalSpace 𝕜
    inst✝⁷ : ContinuousAdd 𝕜
    inst✝⁶ : ContinuousConstSMul 𝕜 𝕜
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Semiring A
    inst✝³ : Algebra 𝕜 A
    inst✝² : Nontrivial 𝕜
    inst✝¹ : T2Space 𝕜
    inst✝ : ContinuousMul 𝕜
    ⊢ IsClosed (Inter.inter (setOf fun a => Eq (a 1) 1) (setOf fun a => ∀ (x y : A …
  -/
  refine IsClosed.inter (isClosed_eq (eval_continuous _) continuous_const) ?_
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝¹⁰ : CommRing 𝕜
    inst✝⁹ : NoZeroDivisors 𝕜
    inst✝⁸ : TopologicalSpace 𝕜
    inst✝⁷ : ContinuousAdd 𝕜
    inst✝⁶ : ContinuousConstSMul 𝕜 𝕜
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : Semiring A
    inst✝³ : Algebra 𝕜 A
    inst✝² : Nontrivial 𝕜
    inst✝¹ : T2Space 𝕜
    inst✝ : ContinuousMul 𝕜
    ⊢ IsClosed (setOf fun a => ∀ (x y : A), Eq (a (HMul.hMul x y)) (HMul.hMul (a x …
  -/
  simpa only [(union_zero 𝕜 A).symm] using union_zero_isClosed _ _
  /-
    🎉 no goals
  -/


theorem apply_mem_spectrum [Nontrivial 𝕜] (φ : characterSpace 𝕜 A) (a : A) : φ a ∈ spectrum 𝕜 a :=
  AlgHom.apply_mem_spectrum φ a


theorem ext_ker {φ ψ : characterSpace 𝕜 A} (h : RingHom.ker φ = RingHom.ker ψ) : φ = ψ := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁷ : CommRing 𝕜
    inst✝⁶ : NoZeroDivisors 𝕜
    inst✝⁵ : TopologicalSpace 𝕜
    inst✝⁴ : ContinuousAdd 𝕜
    inst✝³ : ContinuousConstSMul 𝕜 𝕜
    inst✝² : TopologicalSpace A
    inst✝¹ : Ring A
    inst✝ : Algebra 𝕜 A
    φ ψ : ↑(WeakDual.characterSpace 𝕜 A)
    h : Eq (RingHom.ker φ) (RingHom.ker ψ)
    ⊢ Eq φ ψ
  -/
  ext x
  have : x - algebraMap 𝕜 A (ψ x) ∈ RingHom.ker φ := by
    simpa only [h, RingHom.mem_ker, map_sub, AlgHomClass.commutes] using sub_self (ψ x)
  /-
    case h
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁷ : CommRing 𝕜
    inst✝⁶ : NoZeroDivisors 𝕜
    inst✝⁵ : TopologicalSpace 𝕜
    inst✝⁴ : ContinuousAdd 𝕜
    inst✝³ : ContinuousConstSMul 𝕜 𝕜
    inst✝² : TopologicalSpace A
    inst✝¹ : Ring A
    inst✝ : Algebra 𝕜 A
    φ ψ : ↑(WeakDual.characterSpace 𝕜 A)
    h : Eq (RingHom.ker φ) (RingHom.ker ψ)
    x : A
    this : Membership.mem (RingHom.ker φ) (HSub.hSub x ((algebraMap 𝕜 A) (ψ x)))
    ⊢ Eq (φ x) (ψ x)
  -/
  rwa [RingHom.mem_ker, map_sub, AlgHomClass.commutes, sub_eq_zero] at this
  /-
    🎉 no goals
  -/


/-- The `RingHom.ker` of `φ : characterSpace 𝕜 A` is maximal. -/
instance ker_isMaximal (φ : characterSpace 𝕜 A) : (RingHom.ker φ).IsMaximal :=
  RingHom.ker_isMaximal_of_surjective φ fun z =>
                          /-
                            𝕜 : Type u_1
                            A : Type u_2
                            inst✝⁶ : Field 𝕜
                            inst✝⁵ : TopologicalSpace 𝕜
                            inst✝⁴ : ContinuousAdd 𝕜
                            inst✝³ : ContinuousConstSMul 𝕜 𝕜
                            inst✝² : Ring A
                            inst✝¹ : TopologicalSpace A
                            inst✝ : Algebra 𝕜 A
                            φ : ↑(WeakDual.characterSpace 𝕜 A)
                            z : 𝕜
                            ⊢ Eq (φ ((algebraMap 𝕜 A) z)) z
                          -/
    ⟨algebraMap 𝕜 A z, by simp only [AlgHomClass.commutes, Algebra.id.map_eq_id, RingHom.id_apply]⟩
                          /-
                            🎉 no goals
                          -/


/-- The **Gelfand transform** is an algebra homomorphism (over `𝕜`) from a topological `𝕜`-algebra
`A` into the `𝕜`-algebra of continuous `𝕜`-valued functions on the `characterSpace 𝕜 A`.
The character space itself consists of all algebra homomorphisms from `A` to `𝕜`. -/
@[simps]
def gelfandTransform : A →ₐ[𝕜] C(characterSpace 𝕜 A, 𝕜) where
  toFun a :=
    { toFun := fun φ => φ a
      continuous_toFun := (eval_continuous a).comp continuous_induced_dom }
                 /-
                   𝕜 : Type u_1
                   A : Type u_2
                   inst✝⁶ : CommRing 𝕜
                   inst✝⁵ : NoZeroDivisors 𝕜
                   inst✝⁴ : TopologicalSpace 𝕜
                   inst✝³ : TopologicalRing 𝕜
                   inst✝² : TopologicalSpace A
                   inst✝¹ : Semiring A
                   inst✝ : Algebra 𝕜 A
                   ⊢ Eq ((fun a => { toFun := fun φ => φ a, continuous_toFun := ⋯ }) 1) 1
                 -/
  map_one' := by ext a; simp only [coe_mk, coe_one, Pi.one_apply, map_one a]
                        /-
                          🎉 no goals
                        -/
                     /-
                       𝕜 : Type u_1
                       A : Type u_2
                       inst✝⁶ : CommRing 𝕜
                       inst✝⁵ : NoZeroDivisors 𝕜
                       inst✝⁴ : TopologicalSpace 𝕜
                       inst✝³ : TopologicalRing 𝕜
                       inst✝² : TopologicalSpace A
                       inst✝¹ : Semiring A
                       inst✝ : Algebra 𝕜 A
                       a b : A
                       ⊢ Eq ({ toFun := fun a => { toFun := fun φ => φ a, continuous_toFun := ⋯ }, ma …
                     -/
  map_mul' a b := by ext; simp only [map_mul, coe_mk, coe_mul, Pi.mul_apply]
                          /-
                            🎉 no goals
                          -/
                  /-
                    𝕜 : Type u_1
                    A : Type u_2
                    inst✝⁶ : CommRing 𝕜
                    inst✝⁵ : NoZeroDivisors 𝕜
                    inst✝⁴ : TopologicalSpace 𝕜
                    inst✝³ : TopologicalRing 𝕜
                    inst✝² : TopologicalSpace A
                    inst✝¹ : Semiring A
                    inst✝ : Algebra 𝕜 A
                    ⊢ Eq ((↑{ toFun := fun a => { toFun := fun φ => φ a, continuous_toFun := ⋯ },  …
                  -/
  map_zero' := by ext; simp only [map_zero, coe_mk, coe_mul, coe_zero, Pi.zero_apply]
                       /-
                         🎉 no goals
                       -/
                     /-
                       𝕜 : Type u_1
                       A : Type u_2
                       inst✝⁶ : CommRing 𝕜
                       inst✝⁵ : NoZeroDivisors 𝕜
                       inst✝⁴ : TopologicalSpace 𝕜
                       inst✝³ : TopologicalRing 𝕜
                       inst✝² : TopologicalSpace A
                       inst✝¹ : Semiring A
                       inst✝ : Algebra 𝕜 A
                       a b : A
                       ⊢ Eq ((↑{ toFun := fun a => { toFun := fun φ => φ a, continuous_toFun := ⋯ },  …
                     -/
  map_add' a b := by ext; simp only [map_add, coe_mk, coe_add, Pi.add_apply]
                          /-
                            🎉 no goals
                          -/
                    /-
                      𝕜 : Type u_1
                      A : Type u_2
                      inst✝⁶ : CommRing 𝕜
                      inst✝⁵ : NoZeroDivisors 𝕜
                      inst✝⁴ : TopologicalSpace 𝕜
                      inst✝³ : TopologicalRing 𝕜
                      inst✝² : TopologicalSpace A
                      inst✝¹ : Semiring A
                      inst✝ : Algebra 𝕜 A
                      k : 𝕜
                      ⊢ Eq ((↑↑{ toFun := fun a => { toFun := fun φ => φ a, continuous_toFun := ⋯ }, …
                    -/
  commutes' k := by ext; simp [AlgHomClass.commutes]
                         /-
                           🎉 no goals
                         -/


