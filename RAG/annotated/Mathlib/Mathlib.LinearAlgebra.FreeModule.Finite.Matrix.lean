private noncomputable def linearMapEquivFun : (M →ₗ[R] N) ≃ₗ[S] ChooseBasisIndex R M → N :=
  (chooseBasis R M).repr.congrLeft N S ≪≫ₗ (Finsupp.lsum S).symm ≪≫ₗ
    LinearEquiv.piCongrRight fun _ ↦ LinearMap.ringLmapEquivSelf R S N


instance Module.Free.linearMap [Module.Free S N] : Module.Free S (M →ₗ[R] N) :=
  Module.Free.of_equiv (linearMapEquivFun R S M N).symm


instance Module.Finite.linearMap [Module.Finite S N] : Module.Finite S (M →ₗ[R] N) :=
  Module.Finite.equiv (linearMapEquivFun R S M N).symm


theorem Module.rank_linearMap :
    Module.rank S (M →ₗ[R] N) = lift.{w} (Module.rank R M) * lift.{v} (Module.rank S N) := by
  rw [(linearMapEquivFun R S M N).rank_eq, rank_fun_eq_lift_mul,
    ← finrank_eq_card_chooseBasisIndex, ← finrank_eq_rank R, lift_natCast]


/-- The finrank of `M →ₗ[R] N` as an `S`-module is `(finrank R M) * (finrank S N)`. -/
theorem Module.finrank_linearMap :
    finrank S (M →ₗ[R] N) = finrank R M * finrank S N := by
  /-
    R : Type u
    S : Type u'
    M : Type v
    N : Type w
    inst✝¹² : Ring R
    inst✝¹¹ : Ring S
    inst✝¹⁰ : AddCommGroup M
    inst✝⁹ : Module R M
    inst✝⁸ : Module.Free R M
    inst✝⁷ : Module.Finite R M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : Module R N
    inst✝⁴ : Module S N
    inst✝³ : SMulCommClass R S N
    inst✝² : StrongRankCondition R
    inst✝¹ : StrongRankCondition S
    inst✝ : Module.Free S N
    ⊢ Eq (Module.finrank S (LinearMap (RingHom.id R) M N)) (HMul.hMul (Module.finr …
  -/
  simp_rw [finrank, rank_linearMap, toNat_mul, toNat_lift]
  /-
    🎉 no goals
  -/


theorem Module.rank_linearMap_self :
    Module.rank S (M →ₗ[R] S) = lift.{u'} (Module.rank R M) := by
  /-
    R : Type u
    S : Type u'
    M : Type v
    inst✝⁹ : Ring R
    inst✝⁸ : Ring S
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : Module.Free R M
    inst✝⁴ : Module.Finite R M
    inst✝³ : StrongRankCondition R
    inst✝² : StrongRankCondition S
    inst✝¹ : Module R S
    inst✝ : SMulCommClass R S S
    ⊢ Eq (Module.rank S (LinearMap (RingHom.id R) M S)) (Cardinal.lift.{u', v} (Mo …
  -/
  rw [rank_linearMap, rank_self, lift_one, mul_one]
  /-
    🎉 no goals
  -/


theorem Module.finrank_linearMap_self : finrank S (M →ₗ[R] S) = finrank R M := by
  /-
    R : Type u
    S : Type u'
    M : Type v
    inst✝⁹ : Ring R
    inst✝⁸ : Ring S
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : Module.Free R M
    inst✝⁴ : Module.Finite R M
    inst✝³ : StrongRankCondition R
    inst✝² : StrongRankCondition S
    inst✝¹ : Module R S
    inst✝ : SMulCommClass R S S
    ⊢ Eq (Module.finrank S (LinearMap (RingHom.id R) M S)) (Module.finrank R M)
  -/
  rw [finrank_linearMap, finrank_self, mul_one]
  /-
    🎉 no goals
  -/


instance Finite.algHom : Finite (M →ₐ[K] L) :=
  (linearIndependent_algHom_toLinearMap K M L).finite


theorem cardinalMk_algHom_le_rank : #(M →ₐ[K] L) ≤ lift.{v} (Module.rank K M) := by
  /-
    K : Type u_1
    M : Type u_2
    L : Type v
    inst✝⁷ : CommRing K
    inst✝⁶ : Ring M
    inst✝⁵ : Algebra K M
    inst✝⁴ : Module.Free K M
    inst✝³ : Module.Finite K M
    inst✝² : CommRing L
    inst✝¹ : IsDomain L
    inst✝ : Algebra K L
    ⊢ LE.le (Cardinal.mk (AlgHom K M L)) (Cardinal.lift.{v, u_2} (Module.rank K M))
  -/
  convert (linearIndependent_algHom_toLinearMap K M L).cardinal_lift_le_rank
    /-
      case h.e'_3
      K : Type u_1
      M : Type u_2
      L : Type v
      inst✝⁷ : CommRing K
      inst✝⁶ : Ring M
      inst✝⁵ : Algebra K M
      inst✝⁴ : Module.Free K M
      inst✝³ : Module.Finite K M
      inst✝² : CommRing L
      inst✝¹ : IsDomain L
      inst✝ : Algebra K L
      ⊢ Eq (Cardinal.mk (AlgHom K M L)) (Cardinal.lift.{max u_2 v, max u_2 v} (Cardi …
    -/
  · rw [lift_id]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_4
      K : Type u_1
      M : Type u_2
      L : Type v
      inst✝⁷ : CommRing K
      inst✝⁶ : Ring M
      inst✝⁵ : Algebra K M
      inst✝⁴ : Module.Free K M
      inst✝³ : Module.Finite K M
      inst✝² : CommRing L
      inst✝¹ : IsDomain L
      inst✝ : Algebra K L
      ⊢ Eq (Cardinal.lift.{v, u_2} (Module.rank K M)) (Cardinal.lift.{max u_2 v, max …
    -/
  · have := Module.nontrivial K L
    /-
      case h.e'_4
      K : Type u_1
      M : Type u_2
      L : Type v
      inst✝⁷ : CommRing K
      inst✝⁶ : Ring M
      inst✝⁵ : Algebra K M
      inst✝⁴ : Module.Free K M
      inst✝³ : Module.Finite K M
      inst✝² : CommRing L
      inst✝¹ : IsDomain L
      inst✝ : Algebra K L
      this : Nontrivial K
      ⊢ Eq (Cardinal.lift.{v, u_2} (Module.rank K M)) (Cardinal.lift.{max u_2 v, max …
    -/
    rw [lift_id, Module.rank_linearMap_self]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-11-10")] alias cardinal_mk_algHom_le_rank := cardinalMk_algHom_le_rank


@[stacks 09HS]
theorem card_algHom_le_finrank : Nat.card (M →ₐ[K] L) ≤ finrank K M := by
  /-
    K : Type u_1
    M : Type u_2
    L : Type v
    inst✝⁷ : CommRing K
    inst✝⁶ : Ring M
    inst✝⁵ : Algebra K M
    inst✝⁴ : Module.Free K M
    inst✝³ : Module.Finite K M
    inst✝² : CommRing L
    inst✝¹ : IsDomain L
    inst✝ : Algebra K L
    ⊢ LE.le (Nat.card (AlgHom K M L)) (Module.finrank K M)
  -/
  convert toNat_le_toNat (cardinalMk_algHom_le_rank K M L) ?_
    /-
      case h.e'_4
      K : Type u_1
      M : Type u_2
      L : Type v
      inst✝⁷ : CommRing K
      inst✝⁶ : Ring M
      inst✝⁵ : Algebra K M
      inst✝⁴ : Module.Free K M
      inst✝³ : Module.Finite K M
      inst✝² : CommRing L
      inst✝¹ : IsDomain L
      inst✝ : Algebra K L
      ⊢ Eq (Module.finrank K M) (Cardinal.toNat (Cardinal.lift.{v, u_2} (Module.rank …
    -/
  · rw [toNat_lift, finrank]
    /-
      🎉 no goals
    -/
    /-
      K : Type u_1
      M : Type u_2
      L : Type v
      inst✝⁷ : CommRing K
      inst✝⁶ : Ring M
      inst✝⁵ : Algebra K M
      inst✝⁴ : Module.Free K M
      inst✝³ : Module.Finite K M
      inst✝² : CommRing L
      inst✝¹ : IsDomain L
      inst✝ : Algebra K L
      ⊢ LT.lt (Cardinal.lift.{v, u_2} (Module.rank K M)) Cardinal.aleph0
    -/
  · rw [lift_lt_aleph0]; have := Module.nontrivial K L; apply Module.rank_lt_aleph0
                                                        /-
                                                          🎉 no goals
                                                        -/


instance Module.Finite.addMonoidHom [Module.Finite ℤ N] : Module.Finite ℤ (M →+ N) :=
  Module.Finite.equiv (addMonoidHomLequivInt ℤ).symm


instance Module.Free.addMonoidHom [Module.Free ℤ N] : Module.Free ℤ (M →+ N) :=
  letI : Module.Free ℤ (M →ₗ[ℤ] N) := Module.Free.linearMap _ _ _ _
  Module.Free.of_equiv (addMonoidHomLequivInt ℤ).symm


theorem Matrix.rank_vecMulVec {K m n : Type u} [CommRing K] [Fintype n]
    [DecidableEq n] (w : m → K) (v : n → K) : (Matrix.vecMulVec w v).toLin'.rank ≤ 1 := by
  /-
    K m n : Type u
    inst✝² : CommRing K
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    w : m → K
    v : n → K
    ⊢ LE.le (Matrix.toLin' (Matrix.vecMulVec w v)).rank 1
  -/
  nontriviality K
  /-
    K m n : Type u
    inst✝² : CommRing K
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    w : m → K
    v : n → K
    a✝ : Nontrivial K
    ⊢ LE.le (Matrix.toLin' (Matrix.vecMulVec w v)).rank 1
  -/
  rw [Matrix.vecMulVec_eq (Fin 1), Matrix.toLin'_mul]
  /-
    K m n : Type u
    inst✝² : CommRing K
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    w : m → K
    v : n → K
    a✝ : Nontrivial K
    ⊢ LE.le ((Matrix.toLin' (Matrix.col (Fin 1) w)).comp (Matrix.toLin' (Matrix.ro …
  -/
  refine le_trans (LinearMap.rank_comp_le_left _ _) ?_
  /-
    K m n : Type u
    inst✝² : CommRing K
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    w : m → K
    v : n → K
    a✝ : Nontrivial K
    ⊢ LE.le (Matrix.toLin' (Matrix.col (Fin 1) w)).rank 1
  -/
  refine (LinearMap.rank_le_domain _).trans_eq ?_
  /-
    K m n : Type u
    inst✝² : CommRing K
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    w : m → K
    v : n → K
    a✝ : Nontrivial K
    ⊢ Eq (Module.rank K (Fin 1 → K)) 1
  -/
  rw [rank_fun', Fintype.card_ofSubsingleton, Nat.cast_one]
  /-
    🎉 no goals
  -/

