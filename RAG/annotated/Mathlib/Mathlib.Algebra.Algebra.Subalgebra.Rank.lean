theorem rank_sup_eq_rank_left_mul_rank_of_free :
    Module.rank R ↥(A ⊔ B) = Module.rank R A * Module.rank A (Algebra.adjoin A (B : Set S)) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    A B : Subalgebra R S
    inst✝¹ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝ : Module.Free (Subtype fun x => Membership.mem A x) (Subtype fun x => Me …
    ⊢ Eq (Module.rank R (Subtype fun x => Membership.mem (Max.max A B) x)) (HMul.h …
  -/
  rcases subsingleton_or_nontrivial R with _ | _
    /-
      case inl
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      A B : Subalgebra R S
      inst✝¹ : Module.Free R (Subtype fun x => Membership.mem A x)
      inst✝ : Module.Free (Subtype fun x => Membership.mem A x) (Subtype fun x => Me …
      h✝ : Subsingleton R
      ⊢ Eq (Module.rank R (Subtype fun x => Membership.mem (Max.max A B) x)) (HMul.h …
    -/
  · haveI := Module.subsingleton R S; simp
                                      /-
                                        🎉 no goals
                                      -/
  /-
    case inr
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    A B : Subalgebra R S
    inst✝¹ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝ : Module.Free (Subtype fun x => Membership.mem A x) (Subtype fun x => Me …
    h✝ : Nontrivial R
    ⊢ Eq (Module.rank R (Subtype fun x => Membership.mem (Max.max A B) x)) (HMul.h …
  -/
  nontriviality S using rank_subsingleton'
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    A B : Subalgebra R S
    inst✝¹ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝ : Module.Free (Subtype fun x => Membership.mem A x) (Subtype fun x => Me …
    h✝ : Nontrivial R
    a✝ : Nontrivial S
    ⊢ Eq (Module.rank R (Subtype fun x => Membership.mem (Max.max A B) x)) (HMul.h …
  -/
  letI : Algebra A (Algebra.adjoin A (B : Set S)) := Subalgebra.algebra _
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    A B : Subalgebra R S
    inst✝¹ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝ : Module.Free (Subtype fun x => Membership.mem A x) (Subtype fun x => Me …
    h✝ : Nontrivial R
    a✝ : Nontrivial S
    this : Algebra (Subtype fun x => Membership.mem A x) (Subtype fun x => Members …
    ⊢ Eq (Module.rank R (Subtype fun x => Membership.mem (Max.max A B) x)) (HMul.h …
  -/
  letI : SMul A (Algebra.adjoin A (B : Set S)) := Algebra.toSMul
  haveI : IsScalarTower R A (Algebra.adjoin A (B : Set S)) :=
    IsScalarTower.of_algebraMap_eq (congrFun rfl)
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    A B : Subalgebra R S
    inst✝¹ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝ : Module.Free (Subtype fun x => Membership.mem A x) (Subtype fun x => Me …
    h✝ : Nontrivial R
    a✝ : Nontrivial S
    this✝¹ : Algebra (Subtype fun x => Membership.mem A x) (Subtype fun x => Membe …
    this✝ : SMul (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    this : IsScalarTower R (Subtype fun x => Membership.mem A x) (Subtype fun x => …
    ⊢ Eq (Module.rank R (Subtype fun x => Membership.mem (Max.max A B) x)) (HMul.h …
  -/
  rw [rank_mul_rank R A (Algebra.adjoin A (B : Set S))]
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    A B : Subalgebra R S
    inst✝¹ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝ : Module.Free (Subtype fun x => Membership.mem A x) (Subtype fun x => Me …
    h✝ : Nontrivial R
    a✝ : Nontrivial S
    this✝¹ : Algebra (Subtype fun x => Membership.mem A x) (Subtype fun x => Membe …
    this✝ : SMul (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    this : IsScalarTower R (Subtype fun x => Membership.mem A x) (Subtype fun x => …
    ⊢ Eq (Module.rank R (Subtype fun x => Membership.mem (Max.max A B) x)) (Module …
  -/
  change _ = Module.rank R ((Algebra.adjoin A (B : Set S)).restrictScalars R)
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    A B : Subalgebra R S
    inst✝¹ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝ : Module.Free (Subtype fun x => Membership.mem A x) (Subtype fun x => Me …
    h✝ : Nontrivial R
    a✝ : Nontrivial S
    this✝¹ : Algebra (Subtype fun x => Membership.mem A x) (Subtype fun x => Membe …
    this✝ : SMul (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    this : IsScalarTower R (Subtype fun x => Membership.mem A x) (Subtype fun x => …
    ⊢ Eq (Module.rank R (Subtype fun x => Membership.mem (Max.max A B) x)) (Module …
  -/
  rw [Algebra.restrictScalars_adjoin]; rfl
                                       /-
                                         🎉 no goals
                                       -/


theorem finrank_sup_eq_finrank_left_mul_finrank_of_free :
    finrank R ↥(A ⊔ B) = finrank R A * finrank A (Algebra.adjoin A (B : Set S)) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    A B : Subalgebra R S
    inst✝¹ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝ : Module.Free (Subtype fun x => Membership.mem A x) (Subtype fun x => Me …
    ⊢ Eq (Module.finrank R (Subtype fun x => Membership.mem (Max.max A B) x)) (HMu …
  -/
  simpa only [map_mul] using congr(Cardinal.toNat $(rank_sup_eq_rank_left_mul_rank_of_free A B))
  /-
    🎉 no goals
  -/


theorem finrank_left_dvd_finrank_sup_of_free :
    finrank R A ∣ finrank R ↥(A ⊔ B) := ⟨_, finrank_sup_eq_finrank_left_mul_finrank_of_free A B⟩


theorem rank_sup_eq_rank_right_mul_rank_of_free :
    Module.rank R ↥(A ⊔ B) = Module.rank R B * Module.rank B (Algebra.adjoin B (A : Set S)) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    A B : Subalgebra R S
    inst✝¹ : Module.Free R (Subtype fun x => Membership.mem B x)
    inst✝ : Module.Free (Subtype fun x => Membership.mem B x) (Subtype fun x => Me …
    ⊢ Eq (Module.rank R (Subtype fun x => Membership.mem (Max.max A B) x)) (HMul.h …
  -/
  rw [sup_comm, rank_sup_eq_rank_left_mul_rank_of_free]
  /-
    🎉 no goals
  -/


theorem finrank_sup_eq_finrank_right_mul_finrank_of_free :
    finrank R ↥(A ⊔ B) = finrank R B * finrank B (Algebra.adjoin B (A : Set S)) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    A B : Subalgebra R S
    inst✝¹ : Module.Free R (Subtype fun x => Membership.mem B x)
    inst✝ : Module.Free (Subtype fun x => Membership.mem B x) (Subtype fun x => Me …
    ⊢ Eq (Module.finrank R (Subtype fun x => Membership.mem (Max.max A B) x)) (HMu …
  -/
  rw [sup_comm, finrank_sup_eq_finrank_left_mul_finrank_of_free]
  /-
    🎉 no goals
  -/


theorem finrank_right_dvd_finrank_sup_of_free :
    finrank R B ∣ finrank R ↥(A ⊔ B) := ⟨_, finrank_sup_eq_finrank_right_mul_finrank_of_free A B⟩


