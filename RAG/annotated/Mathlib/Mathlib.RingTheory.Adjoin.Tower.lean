theorem adjoin_restrictScalars (C D E : Type*) [CommSemiring C] [CommSemiring D] [CommSemiring E]
    [Algebra C D] [Algebra C E] [Algebra D E] [IsScalarTower C D E] (S : Set E) :
    (Algebra.adjoin D S).restrictScalars C =
      (Algebra.adjoin ((⊤ : Subalgebra C D).map (IsScalarTower.toAlgHom C D E)) S).restrictScalars
        C := by
  suffices
    Set.range (algebraMap D E) =
      Set.range (algebraMap ((⊤ : Subalgebra C D).map (IsScalarTower.toAlgHom C D E)) E) by
    ext x
    change x ∈ Subsemiring.closure (_ ∪ S) ↔ x ∈ Subsemiring.closure (_ ∪ S)
    rw [this]
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝⁶ : CommSemiring C
    inst✝⁵ : CommSemiring D
    inst✝⁴ : CommSemiring E
    inst✝³ : Algebra C D
    inst✝² : Algebra C E
    inst✝¹ : Algebra D E
    inst✝ : IsScalarTower C D E
    S : Set E
    ⊢ Eq (Set.range ⇑(algebraMap D E)) (Set.range ⇑(algebraMap (Subtype fun x => M …
  -/
  ext x
  /-
    case h
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝⁶ : CommSemiring C
    inst✝⁵ : CommSemiring D
    inst✝⁴ : CommSemiring E
    inst✝³ : Algebra C D
    inst✝² : Algebra C E
    inst✝¹ : Algebra D E
    inst✝ : IsScalarTower C D E
    S : Set E
    x : E
    ⊢ Iff (Membership.mem (Set.range ⇑(algebraMap D E)) x) (Membership.mem (Set.ra …
  -/
  constructor
    /-
      case h.mp
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝⁶ : CommSemiring C
      inst✝⁵ : CommSemiring D
      inst✝⁴ : CommSemiring E
      inst✝³ : Algebra C D
      inst✝² : Algebra C E
      inst✝¹ : Algebra D E
      inst✝ : IsScalarTower C D E
      S : Set E
      x : E
      ⊢ Membership.mem (Set.range ⇑(algebraMap D E)) x → Membership.mem (Set.range ⇑ …
    -/
  · rintro ⟨y, hy⟩
    /-
      case h.mp.intro
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝⁶ : CommSemiring C
      inst✝⁵ : CommSemiring D
      inst✝⁴ : CommSemiring E
      inst✝³ : Algebra C D
      inst✝² : Algebra C E
      inst✝¹ : Algebra D E
      inst✝ : IsScalarTower C D E
      S : Set E
      x : E
      y : D
      hy : Eq ((algebraMap D E) y) x
      ⊢ Membership.mem (Set.range ⇑(algebraMap (Subtype fun x => Membership.mem (Sub …
    -/
    exact ⟨⟨algebraMap D E y, ⟨y, ⟨Algebra.mem_top, rfl⟩⟩⟩, hy⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝⁶ : CommSemiring C
      inst✝⁵ : CommSemiring D
      inst✝⁴ : CommSemiring E
      inst✝³ : Algebra C D
      inst✝² : Algebra C E
      inst✝¹ : Algebra D E
      inst✝ : IsScalarTower C D E
      S : Set E
      x : E
      ⊢ Membership.mem (Set.range ⇑(algebraMap (Subtype fun x => Membership.mem (Sub …
    -/
  · rintro ⟨⟨y, ⟨z, ⟨h0, h1⟩⟩⟩, h2⟩
    /-
      case h.mpr.intro.mk.intro.intro
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝⁶ : CommSemiring C
      inst✝⁵ : CommSemiring D
      inst✝⁴ : CommSemiring E
      inst✝³ : Algebra C D
      inst✝² : Algebra C E
      inst✝¹ : Algebra D E
      inst✝ : IsScalarTower C D E
      S : Set E
      x y : E
      z : D
      h0 : Membership.mem (↑Top.top.toSubsemiring) z
      h1 : Eq (↑(IsScalarTower.toAlgHom C D E) z) y
      h2 : Eq ((algebraMap (Subtype fun x => Membership.mem (Subalgebra.map (IsScala …
      ⊢ Membership.mem (Set.range ⇑(algebraMap D E)) x
    -/
    exact ⟨z, Eq.trans h1 h2⟩
    /-
      🎉 no goals
    -/


theorem adjoin_res_eq_adjoin_res (C D E F : Type*) [CommSemiring C] [CommSemiring D]
    [CommSemiring E] [CommSemiring F] [Algebra C D] [Algebra C E] [Algebra C F] [Algebra D F]
    [Algebra E F] [IsScalarTower C D F] [IsScalarTower C E F] {S : Set D} {T : Set E}
    (hS : Algebra.adjoin C S = ⊤) (hT : Algebra.adjoin C T = ⊤) :
    (Algebra.adjoin E (algebraMap D F '' S)).restrictScalars C =
      (Algebra.adjoin D (algebraMap E F '' T)).restrictScalars C := by
  rw [adjoin_restrictScalars C E, adjoin_restrictScalars C D, ← hS, ← hT, ← Algebra.adjoin_image,
    ← Algebra.adjoin_image, ← AlgHom.coe_toRingHom, ← AlgHom.coe_toRingHom,
    IsScalarTower.coe_toAlgHom, IsScalarTower.coe_toAlgHom, ← adjoin_union_eq_adjoin_adjoin, ←
    adjoin_union_eq_adjoin_adjoin, Set.union_comm]


theorem Algebra.fg_trans' {R S A : Type*} [CommSemiring R] [CommSemiring S] [Semiring A]
    [Algebra R S] [Algebra S A] [Algebra R A] [IsScalarTower R S A] (hRS : (⊤ : Subalgebra R S).FG)
    (hSA : (⊤ : Subalgebra S A).FG) : (⊤ : Subalgebra R A).FG :=
  let ⟨s, hs⟩ := hRS
  let ⟨t, ht⟩ := hSA
  ⟨s.image (algebraMap S A) ∪ t, by
    rw [Finset.coe_union, Finset.coe_image, Algebra.adjoin_algebraMap_image_union_eq_adjoin_adjoin,
      hs, Algebra.adjoin_top, ht, Subalgebra.restrictScalars_top, Subalgebra.restrictScalars_top]⟩


theorem exists_subalgebra_of_fg (hAC : (⊤ : Subalgebra A C).FG) (hBC : (⊤ : Submodule B C).FG) :
    ∃ B₀ : Subalgebra A B, B₀.FG ∧ (⊤ : Submodule B₀ C).FG := by
  /-
    A : Type w
    B : Type u₁
    C : Type u_1
    inst✝⁶ : CommSemiring A
    inst✝⁵ : CommSemiring B
    inst✝⁴ : Semiring C
    inst✝³ : Algebra A B
    inst✝² : Algebra B C
    inst✝¹ : Algebra A C
    inst✝ : IsScalarTower A B C
    hAC : Top.top.FG
    hBC : Top.top.FG
    ⊢ Exists fun B₀ => And B₀.FG Top.top.FG
  -/
  cases' hAC with x hx
  /-
    case intro
    A : Type w
    B : Type u₁
    C : Type u_1
    inst✝⁶ : CommSemiring A
    inst✝⁵ : CommSemiring B
    inst✝⁴ : Semiring C
    inst✝³ : Algebra A B
    inst✝² : Algebra B C
    inst✝¹ : Algebra A C
    inst✝ : IsScalarTower A B C
    hBC : Top.top.FG
    x : Finset C
    hx : Eq (Algebra.adjoin A ↑x) Top.top
    ⊢ Exists fun B₀ => And B₀.FG Top.top.FG
  -/
  cases' hBC with y hy
  /-
    case intro.intro
    A : Type w
    B : Type u₁
    C : Type u_1
    inst✝⁶ : CommSemiring A
    inst✝⁵ : CommSemiring B
    inst✝⁴ : Semiring C
    inst✝³ : Algebra A B
    inst✝² : Algebra B C
    inst✝¹ : Algebra A C
    inst✝ : IsScalarTower A B C
    x : Finset C
    hx : Eq (Algebra.adjoin A ↑x) Top.top
    y : Finset C
    hy : Eq (Submodule.span B ↑y) Top.top
    ⊢ Exists fun B₀ => And B₀.FG Top.top.FG
  -/
  have := hy
  /-
    case intro.intro
    A : Type w
    B : Type u₁
    C : Type u_1
    inst✝⁶ : CommSemiring A
    inst✝⁵ : CommSemiring B
    inst✝⁴ : Semiring C
    inst✝³ : Algebra A B
    inst✝² : Algebra B C
    inst✝¹ : Algebra A C
    inst✝ : IsScalarTower A B C
    x : Finset C
    hx : Eq (Algebra.adjoin A ↑x) Top.top
    y : Finset C
    hy this : Eq (Submodule.span B ↑y) Top.top
    ⊢ Exists fun B₀ => And B₀.FG Top.top.FG
  -/
  simp_rw [eq_top_iff', mem_span_finset] at this
  /-
    case intro.intro
    A : Type w
    B : Type u₁
    C : Type u_1
    inst✝⁶ : CommSemiring A
    inst✝⁵ : CommSemiring B
    inst✝⁴ : Semiring C
    inst✝³ : Algebra A B
    inst✝² : Algebra B C
    inst✝¹ : Algebra A C
    inst✝ : IsScalarTower A B C
    x : Finset C
    hx : Eq (Algebra.adjoin A ↑x) Top.top
    y : Finset C
    hy : Eq (Submodule.span B ↑y) Top.top
    this : ∀ (x : C), Exists fun f => Eq (y.sum fun i => HSMul.hSMul (f i) i) x
    ⊢ Exists fun B₀ => And B₀.FG Top.top.FG
  -/
  choose f hf using this
  /-
    case intro.intro
    A : Type w
    B : Type u₁
    C : Type u_1
    inst✝⁶ : CommSemiring A
    inst✝⁵ : CommSemiring B
    inst✝⁴ : Semiring C
    inst✝³ : Algebra A B
    inst✝² : Algebra B C
    inst✝¹ : Algebra A C
    inst✝ : IsScalarTower A B C
    x : Finset C
    hx : Eq (Algebra.adjoin A ↑x) Top.top
    y : Finset C
    hy : Eq (Submodule.span B ↑y) Top.top
    f : C → C → B
    hf : ∀ (x : C), Eq (y.sum fun i => HSMul.hSMul (f x i) i) x
    ⊢ Exists fun B₀ => And B₀.FG Top.top.FG
  -/
  let s : Finset B := Finset.image₂ f (x ∪ y * y) y
  have hxy :
    ∀ xi ∈ x, xi ∈ span (Algebra.adjoin A (↑s : Set B)) (↑(insert 1 y : Finset C) : Set C) :=
    fun xi hxi =>
    hf xi ▸
      sum_mem fun yj hyj =>
        smul_mem (span (Algebra.adjoin A (↑s : Set B)) (↑(insert 1 y : Finset C) : Set C))
          ⟨f xi yj, Algebra.subset_adjoin <| mem_image₂_of_mem (mem_union_left _ hxi) hyj⟩
          (subset_span <| mem_insert_of_mem hyj)
  have hyy :
    span (Algebra.adjoin A (↑s : Set B)) (↑(insert 1 y : Finset C) : Set C) *
        span (Algebra.adjoin A (↑s : Set B)) (↑(insert 1 y : Finset C) : Set C) ≤
      span (Algebra.adjoin A (↑s : Set B)) (↑(insert 1 y : Finset C) : Set C) := by
    rw [span_mul_span, span_le, coe_insert]
    rintro _ ⟨yi, rfl | hyi, yj, rfl | hyj, rfl⟩ <;> dsimp
    · rw [mul_one]
      exact subset_span (Set.mem_insert _ _)
    · rw [one_mul]
      exact subset_span (Set.mem_insert_of_mem _ hyj)
    · rw [mul_one]
      exact subset_span (Set.mem_insert_of_mem _ hyi)
    · rw [← hf (yi * yj)]
      exact
        SetLike.mem_coe.2
          (sum_mem fun yk hyk =>
            smul_mem (span (Algebra.adjoin A (↑s : Set B)) (insert 1 ↑y : Set C))
              ⟨f (yi * yj) yk,
                Algebra.subset_adjoin <|
                  mem_image₂_of_mem (mem_union_right _ <| mul_mem_mul hyi hyj) hyk⟩
              (subset_span <| Set.mem_insert_of_mem _ hyk : yk ∈ _))
  /-
    case intro.intro
    A : Type w
    B : Type u₁
    C : Type u_1
    inst✝⁶ : CommSemiring A
    inst✝⁵ : CommSemiring B
    inst✝⁴ : Semiring C
    inst✝³ : Algebra A B
    inst✝² : Algebra B C
    inst✝¹ : Algebra A C
    inst✝ : IsScalarTower A B C
    x : Finset C
    hx : Eq (Algebra.adjoin A ↑x) Top.top
    y : Finset C
    hy : Eq (Submodule.span B ↑y) Top.top
    f : C → C → B
    hf : ∀ (x : C), Eq (y.sum fun i => HSMul.hSMul (f x i) i) x
    s : Finset B := Finset.image₂ f (Union.union x (HMul.hMul y y)) y
    hxy : ∀ (xi : C), Membership.mem x xi → Membership.mem (Submodule.span (Subtyp …
    hyy : LE.le (HMul.hMul (Submodule.span (Subtype fun x => Membership.mem (Algeb …
    ⊢ Exists fun B₀ => And B₀.FG Top.top.FG
  -/
  refine ⟨Algebra.adjoin A (↑s : Set B), Subalgebra.fg_adjoin_finset _, insert 1 y, ?_⟩
  /-
    case intro.intro
    A : Type w
    B : Type u₁
    C : Type u_1
    inst✝⁶ : CommSemiring A
    inst✝⁵ : CommSemiring B
    inst✝⁴ : Semiring C
    inst✝³ : Algebra A B
    inst✝² : Algebra B C
    inst✝¹ : Algebra A C
    inst✝ : IsScalarTower A B C
    x : Finset C
    hx : Eq (Algebra.adjoin A ↑x) Top.top
    y : Finset C
    hy : Eq (Submodule.span B ↑y) Top.top
    f : C → C → B
    hf : ∀ (x : C), Eq (y.sum fun i => HSMul.hSMul (f x i) i) x
    s : Finset B := Finset.image₂ f (Union.union x (HMul.hMul y y)) y
    hxy : ∀ (xi : C), Membership.mem x xi → Membership.mem (Submodule.span (Subtyp …
    hyy : LE.le (HMul.hMul (Submodule.span (Subtype fun x => Membership.mem (Algeb …
    ⊢ Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin A ↑s) x) …
  -/
  convert restrictScalars_injective A (Algebra.adjoin A (s : Set B)) C _
  rw [restrictScalars_top, eq_top_iff, ← Algebra.top_toSubmodule, ← hx, Algebra.adjoin_eq_span,
    span_le]
  refine fun r hr =>
    Submonoid.closure_induction (fun c hc => hxy c hc) (subset_span <| mem_insert_self _ _)
      (fun p q _ _ hp hq => hyy <| Submodule.mul_mem_mul hp hq) hr


/-- **Artin--Tate lemma**: if A ⊆ B ⊆ C is a chain of subrings of commutative rings, and
A is noetherian, and C is algebra-finite over A, and C is module-finite over B,
then B is algebra-finite over A.

References: Atiyah--Macdonald Proposition 7.8; Stacks 00IS; Altman--Kleiman 16.17. -/
theorem fg_of_fg_of_fg [IsNoetherianRing A] (hAC : (⊤ : Subalgebra A C).FG)
    (hBC : (⊤ : Submodule B C).FG) (hBCi : Function.Injective (algebraMap B C)) :
    (⊤ : Subalgebra A B).FG :=
  let ⟨B₀, hAB₀, hB₀C⟩ := exists_subalgebra_of_fg A B C hAC hBC
  Algebra.fg_trans' (B₀.fg_top.2 hAB₀) <|
    Subalgebra.fg_of_submodule_fg <|
      have : IsNoetherianRing B₀ := isNoetherianRing_of_fg hAB₀
      have : Module.Finite B₀ C := ⟨hB₀C⟩
      fg_of_injective (IsScalarTower.toAlgHom B₀ B C).toLinearMap hBCi


