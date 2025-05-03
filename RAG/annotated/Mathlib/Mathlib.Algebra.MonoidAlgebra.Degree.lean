theorem sup_support_add_le :
    (f + g).support.sup degb ≤ f.support.sup degb ⊔ g.support.sup degb := by
  classical
  exact (Finset.sup_mono Finsupp.support_add).trans_eq Finset.sup_union


theorem le_inf_support_add : f.support.inf degt ⊓ g.support.inf degt ≤ (f + g).support.inf degt :=
  sup_support_add_le (fun a : A => OrderDual.toDual (degt a)) f g


theorem sup_support_mul_le {degb : A → B} (degbm : ∀ {a b}, degb (a + b) ≤ degb a + degb b)
    (f g : R[A]) :
    (f * g).support.sup degb ≤ f.support.sup degb + g.support.sup degb := by
  classical
  exact (Finset.sup_mono <| support_mul _ _).trans <| Finset.sup_add_le.2 fun _fd fds _gd gds ↦
    degbm.trans <| add_le_add (Finset.le_sup fds) (Finset.le_sup gds)


theorem le_inf_support_mul {degt : A → T} (degtm : ∀ {a b}, degt a + degt b ≤ degt (a + b))
    (f g : R[A]) :
    f.support.inf degt + g.support.inf degt ≤ (f * g).support.inf degt :=
  sup_support_mul_le (B := Tᵒᵈ) degtm f g


theorem sup_support_list_prod_le (degb0 : degb 0 ≤ 0)
    (degbm : ∀ a b, degb (a + b) ≤ degb a + degb b) :
    ∀ l : List R[A],
      l.prod.support.sup degb ≤ (l.map fun f : R[A] => f.support.sup degb).sum
  | [] => by
    /-
      R : Type u_1
      A : Type u_3
      B : Type u_5
      inst✝⁶ : SemilatticeSup B
      inst✝⁵ : OrderBot B
      inst✝⁴ : Semiring R
      inst✝³ : AddMonoid A
      inst✝² : AddMonoid B
      inst✝¹ : AddLeftMono B
      inst✝ : AddRightMono B
      degb : A → B
      degb0 : LE.le (degb 0) 0
      degbm : ∀ (a b : A), LE.le (degb (HAdd.hAdd a b)) (HAdd.hAdd (degb a) (degb b))
      ⊢ LE.le (List.nil.prod.support.sup degb) (List.map (fun f => f.support.sup deg …
    -/
    rw [List.map_nil, Finset.sup_le_iff, List.prod_nil, List.sum_nil]
    /-
      R : Type u_1
      A : Type u_3
      B : Type u_5
      inst✝⁶ : SemilatticeSup B
      inst✝⁵ : OrderBot B
      inst✝⁴ : Semiring R
      inst✝³ : AddMonoid A
      inst✝² : AddMonoid B
      inst✝¹ : AddLeftMono B
      inst✝ : AddRightMono B
      degb : A → B
      degb0 : LE.le (degb 0) 0
      degbm : ∀ (a b : A), LE.le (degb (HAdd.hAdd a b)) (HAdd.hAdd (degb a) (degb b))
      ⊢ ∀ (b : A), Membership.mem (Finsupp.support 1) b → LE.le (degb b) 0
    -/
    exact fun a ha => by rwa [Finset.mem_singleton.mp (Finsupp.support_single_subset ha)]
    /-
      🎉 no goals
    -/
  | f::fs => by
    /-
      R : Type u_1
      A : Type u_3
      B : Type u_5
      inst✝⁶ : SemilatticeSup B
      inst✝⁵ : OrderBot B
      inst✝⁴ : Semiring R
      inst✝³ : AddMonoid A
      inst✝² : AddMonoid B
      inst✝¹ : AddLeftMono B
      inst✝ : AddRightMono B
      degb : A → B
      degb0 : LE.le (degb 0) 0
      degbm : ∀ (a b : A), LE.le (degb (HAdd.hAdd a b)) (HAdd.hAdd (degb a) (degb b))
      f : AddMonoidAlgebra R A
      fs : List (AddMonoidAlgebra R A)
      ⊢ LE.le ((List.cons f fs).prod.support.sup degb) (List.map (fun f => f.support …
    -/
    rw [List.prod_cons, List.map_cons, List.sum_cons]
    exact (sup_support_mul_le (@fun a b => degbm a b) _ _).trans
        (add_le_add_left (sup_support_list_prod_le degb0 degbm fs) _)


theorem le_inf_support_list_prod (degt0 : 0 ≤ degt 0)
    (degtm : ∀ a b, degt a + degt b ≤ degt (a + b)) (l : List R[A]) :
    (l.map fun f : R[A] => f.support.inf degt).sum ≤ l.prod.support.inf degt := by
  /-
    R : Type u_1
    A : Type u_3
    T : Type u_4
    inst✝⁶ : SemilatticeInf T
    inst✝⁵ : OrderTop T
    inst✝⁴ : Semiring R
    inst✝³ : AddMonoid A
    inst✝² : AddMonoid T
    inst✝¹ : AddLeftMono T
    inst✝ : AddRightMono T
    degt : A → T
    degt0 : LE.le 0 (degt 0)
    degtm : ∀ (a b : A), LE.le (HAdd.hAdd (degt a) (degt b)) (degt (HAdd.hAdd a b))
    l : List (AddMonoidAlgebra R A)
    ⊢ LE.le (List.map (fun f => f.support.inf degt) l).sum (l.prod.support.inf degt)
  -/
  refine OrderDual.ofDual_le_ofDual.mpr ?_
  /-
    R : Type u_1
    A : Type u_3
    T : Type u_4
    inst✝⁶ : SemilatticeInf T
    inst✝⁵ : OrderTop T
    inst✝⁴ : Semiring R
    inst✝³ : AddMonoid A
    inst✝² : AddMonoid T
    inst✝¹ : AddLeftMono T
    inst✝ : AddRightMono T
    degt : A → T
    degt0 : LE.le 0 (degt 0)
    degtm : ∀ (a b : A), LE.le (HAdd.hAdd (degt a) (degt b)) (degt (HAdd.hAdd a b))
    l : List (AddMonoidAlgebra R A)
    ⊢ LE.le (Quot.lift (fun l => List.foldr (fun x1 x2 => Min.min x1 x2) Top.top l …
  -/
  refine sup_support_list_prod_le ?_ ?_ l
    /-
      case refine_1
      R : Type u_1
      A : Type u_3
      T : Type u_4
      inst✝⁶ : SemilatticeInf T
      inst✝⁵ : OrderTop T
      inst✝⁴ : Semiring R
      inst✝³ : AddMonoid A
      inst✝² : AddMonoid T
      inst✝¹ : AddLeftMono T
      inst✝ : AddRightMono T
      degt : A → T
      degt0 : LE.le 0 (degt 0)
      degtm : ∀ (a b : A), LE.le (HAdd.hAdd (degt a) (degt b)) (degt (HAdd.hAdd a b))
      l : List (AddMonoidAlgebra R A)
      ⊢ LE.le (degt 0) 0
    -/
  · refine (OrderDual.ofDual_le_ofDual.mp ?_)
    /-
      case refine_1
      R : Type u_1
      A : Type u_3
      T : Type u_4
      inst✝⁶ : SemilatticeInf T
      inst✝⁵ : OrderTop T
      inst✝⁴ : Semiring R
      inst✝³ : AddMonoid A
      inst✝² : AddMonoid T
      inst✝¹ : AddLeftMono T
      inst✝ : AddRightMono T
      degt : A → T
      degt0 : LE.le 0 (degt 0)
      degtm : ∀ (a b : A), LE.le (HAdd.hAdd (degt a) (degt b)) (degt (HAdd.hAdd a b))
      l : List (AddMonoidAlgebra R A)
      ⊢ LE.le (OrderDual.ofDual 0) (OrderDual.ofDual (degt 0))
    -/
    exact degt0
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      A : Type u_3
      T : Type u_4
      inst✝⁶ : SemilatticeInf T
      inst✝⁵ : OrderTop T
      inst✝⁴ : Semiring R
      inst✝³ : AddMonoid A
      inst✝² : AddMonoid T
      inst✝¹ : AddLeftMono T
      inst✝ : AddRightMono T
      degt : A → T
      degt0 : LE.le 0 (degt 0)
      degtm : ∀ (a b : A), LE.le (HAdd.hAdd (degt a) (degt b)) (degt (HAdd.hAdd a b))
      l : List (AddMonoidAlgebra R A)
      ⊢ ∀ (a b : A), LE.le (degt (HAdd.hAdd a b)) (HAdd.hAdd (degt a) (degt b))
    -/
  · refine (fun a b => OrderDual.ofDual_le_ofDual.mp ?_)
    /-
      case refine_2
      R : Type u_1
      A : Type u_3
      T : Type u_4
      inst✝⁶ : SemilatticeInf T
      inst✝⁵ : OrderTop T
      inst✝⁴ : Semiring R
      inst✝³ : AddMonoid A
      inst✝² : AddMonoid T
      inst✝¹ : AddLeftMono T
      inst✝ : AddRightMono T
      degt : A → T
      degt0 : LE.le 0 (degt 0)
      degtm : ∀ (a b : A), LE.le (HAdd.hAdd (degt a) (degt b)) (degt (HAdd.hAdd a b))
      l : List (AddMonoidAlgebra R A)
      a b : A
      ⊢ LE.le (OrderDual.ofDual (HAdd.hAdd (degt a) (degt b))) (OrderDual.ofDual (de …
    -/
    exact degtm a b
    /-
      🎉 no goals
    -/


theorem sup_support_pow_le (degb0 : degb 0 ≤ 0) (degbm : ∀ a b, degb (a + b) ≤ degb a + degb b)
    (n : ℕ) (f : R[A]) : (f ^ n).support.sup degb ≤ n • f.support.sup degb := by
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁶ : SemilatticeSup B
    inst✝⁵ : OrderBot B
    inst✝⁴ : Semiring R
    inst✝³ : AddMonoid A
    inst✝² : AddMonoid B
    inst✝¹ : AddLeftMono B
    inst✝ : AddRightMono B
    degb : A → B
    degb0 : LE.le (degb 0) 0
    degbm : ∀ (a b : A), LE.le (degb (HAdd.hAdd a b)) (HAdd.hAdd (degb a) (degb b))
    n : Nat
    f : AddMonoidAlgebra R A
    ⊢ LE.le ((HPow.hPow f n).support.sup degb) (HSMul.hSMul n (f.support.sup degb))
  -/
  rw [← List.prod_replicate, ← List.sum_replicate]
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁶ : SemilatticeSup B
    inst✝⁵ : OrderBot B
    inst✝⁴ : Semiring R
    inst✝³ : AddMonoid A
    inst✝² : AddMonoid B
    inst✝¹ : AddLeftMono B
    inst✝ : AddRightMono B
    degb : A → B
    degb0 : LE.le (degb 0) 0
    degbm : ∀ (a b : A), LE.le (degb (HAdd.hAdd a b)) (HAdd.hAdd (degb a) (degb b))
    n : Nat
    f : AddMonoidAlgebra R A
    ⊢ LE.le ((List.replicate n f).prod.support.sup degb) (List.replicate n (f.supp …
  -/
  refine (sup_support_list_prod_le degb0 degbm _).trans_eq ?_
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁶ : SemilatticeSup B
    inst✝⁵ : OrderBot B
    inst✝⁴ : Semiring R
    inst✝³ : AddMonoid A
    inst✝² : AddMonoid B
    inst✝¹ : AddLeftMono B
    inst✝ : AddRightMono B
    degb : A → B
    degb0 : LE.le (degb 0) 0
    degbm : ∀ (a b : A), LE.le (degb (HAdd.hAdd a b)) (HAdd.hAdd (degb a) (degb b))
    n : Nat
    f : AddMonoidAlgebra R A
    ⊢ Eq (List.map (fun f => f.support.sup degb) (List.replicate n f)).sum (List.r …
  -/
  rw [List.map_replicate]
  /-
    🎉 no goals
  -/


theorem le_inf_support_pow (degt0 : 0 ≤ degt 0) (degtm : ∀ a b, degt a + degt b ≤ degt (a + b))
    (n : ℕ) (f : R[A]) : n • f.support.inf degt ≤ (f ^ n).support.inf degt := by
  refine OrderDual.ofDual_le_ofDual.mpr <| sup_support_pow_le (OrderDual.ofDual_le_ofDual.mp ?_)
      (fun a b => OrderDual.ofDual_le_ofDual.mp ?_) n f
    /-
      case refine_1
      R : Type u_1
      A : Type u_3
      T : Type u_4
      inst✝⁶ : SemilatticeInf T
      inst✝⁵ : OrderTop T
      inst✝⁴ : Semiring R
      inst✝³ : AddMonoid A
      inst✝² : AddMonoid T
      inst✝¹ : AddLeftMono T
      inst✝ : AddRightMono T
      degt : A → T
      degt0 : LE.le 0 (degt 0)
      degtm : ∀ (a b : A), LE.le (HAdd.hAdd (degt a) (degt b)) (degt (HAdd.hAdd a b))
      n : Nat
      f : AddMonoidAlgebra R A
      ⊢ LE.le (OrderDual.ofDual (degt 0)) (OrderDual.ofDual 0)
    -/
  · exact degt0
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      A : Type u_3
      T : Type u_4
      inst✝⁶ : SemilatticeInf T
      inst✝⁵ : OrderTop T
      inst✝⁴ : Semiring R
      inst✝³ : AddMonoid A
      inst✝² : AddMonoid T
      inst✝¹ : AddLeftMono T
      inst✝ : AddRightMono T
      degt : A → T
      degt0 : LE.le 0 (degt 0)
      degtm : ∀ (a b : A), LE.le (HAdd.hAdd (degt a) (degt b)) (degt (HAdd.hAdd a b))
      n : Nat
      f : AddMonoidAlgebra R A
      a b : A
      ⊢ LE.le (OrderDual.ofDual (degt (HAdd.hAdd a b))) (OrderDual.ofDual (HAdd.hAdd …
    -/
  · exact degtm _ _
    /-
      🎉 no goals
    -/


theorem sup_support_multiset_prod_le (degb0 : degb 0 ≤ 0)
    (degbm : ∀ a b, degb (a + b) ≤ degb a + degb b) (m : Multiset R[A]) :
    m.prod.support.sup degb ≤ (m.map fun f : R[A] => f.support.sup degb).sum := by
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁶ : SemilatticeSup B
    inst✝⁵ : OrderBot B
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid A
    inst✝² : AddCommMonoid B
    inst✝¹ : AddLeftMono B
    inst✝ : AddRightMono B
    degb : A → B
    degb0 : LE.le (degb 0) 0
    degbm : ∀ (a b : A), LE.le (degb (HAdd.hAdd a b)) (HAdd.hAdd (degb a) (degb b))
    m : Multiset (AddMonoidAlgebra R A)
    ⊢ LE.le (m.prod.support.sup degb) (Multiset.map (fun f => f.support.sup degb)  …
  -/
  induction m using Quot.inductionOn
  /-
    case h
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁶ : SemilatticeSup B
    inst✝⁵ : OrderBot B
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid A
    inst✝² : AddCommMonoid B
    inst✝¹ : AddLeftMono B
    inst✝ : AddRightMono B
    degb : A → B
    degb0 : LE.le (degb 0) 0
    degbm : ∀ (a b : A), LE.le (degb (HAdd.hAdd a b)) (HAdd.hAdd (degb a) (degb b))
    a✝ : List (AddMonoidAlgebra R A)
    ⊢ LE.le ((Multiset.prod (Quot.mk (⇑(List.isSetoid (AddMonoidAlgebra R A))) a✝) …
  -/
  rw [Multiset.quot_mk_to_coe'', Multiset.map_coe, Multiset.sum_coe, Multiset.prod_coe]
  /-
    case h
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁶ : SemilatticeSup B
    inst✝⁵ : OrderBot B
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid A
    inst✝² : AddCommMonoid B
    inst✝¹ : AddLeftMono B
    inst✝ : AddRightMono B
    degb : A → B
    degb0 : LE.le (degb 0) 0
    degbm : ∀ (a b : A), LE.le (degb (HAdd.hAdd a b)) (HAdd.hAdd (degb a) (degb b))
    a✝ : List (AddMonoidAlgebra R A)
    ⊢ LE.le (a✝.prod.support.sup degb) (List.map (fun f => f.support.sup degb) a✝) …
  -/
  exact sup_support_list_prod_le degb0 degbm _
  /-
    🎉 no goals
  -/


theorem le_inf_support_multiset_prod (degt0 : 0 ≤ degt 0)
    (degtm : ∀ a b, degt a + degt b ≤ degt (a + b)) (m : Multiset R[A]) :
    (m.map fun f : R[A] => f.support.inf degt).sum ≤ m.prod.support.inf degt := by
  refine OrderDual.ofDual_le_ofDual.mpr <|
    sup_support_multiset_prod_le (OrderDual.ofDual_le_ofDual.mp ?_)
      (fun a b => OrderDual.ofDual_le_ofDual.mp ?_) m
    /-
      case refine_1
      R : Type u_1
      A : Type u_3
      T : Type u_4
      inst✝⁶ : SemilatticeInf T
      inst✝⁵ : OrderTop T
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid A
      inst✝² : AddCommMonoid T
      inst✝¹ : AddLeftMono T
      inst✝ : AddRightMono T
      degt : A → T
      degt0 : LE.le 0 (degt 0)
      degtm : ∀ (a b : A), LE.le (HAdd.hAdd (degt a) (degt b)) (degt (HAdd.hAdd a b))
      m : Multiset (AddMonoidAlgebra R A)
      ⊢ LE.le (OrderDual.ofDual (degt 0)) (OrderDual.ofDual 0)
    -/
  · exact degt0
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      A : Type u_3
      T : Type u_4
      inst✝⁶ : SemilatticeInf T
      inst✝⁵ : OrderTop T
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid A
      inst✝² : AddCommMonoid T
      inst✝¹ : AddLeftMono T
      inst✝ : AddRightMono T
      degt : A → T
      degt0 : LE.le 0 (degt 0)
      degtm : ∀ (a b : A), LE.le (HAdd.hAdd (degt a) (degt b)) (degt (HAdd.hAdd a b))
      m : Multiset (AddMonoidAlgebra R A)
      a b : A
      ⊢ LE.le (OrderDual.ofDual (degt (HAdd.hAdd a b))) (OrderDual.ofDual (HAdd.hAdd …
    -/
  · exact degtm _ _
    /-
      🎉 no goals
    -/


theorem sup_support_finset_prod_le (degb0 : degb 0 ≤ 0)
    (degbm : ∀ a b, degb (a + b) ≤ degb a + degb b) (s : Finset ι) (f : ι → R[A]) :
    (∏ i ∈ s, f i).support.sup degb ≤ ∑ i ∈ s, (f i).support.sup degb :=
  (sup_support_multiset_prod_le degb0 degbm _).trans_eq <| congr_arg _ <| Multiset.map_map _ _ _


theorem le_inf_support_finset_prod (degt0 : 0 ≤ degt 0)
    (degtm : ∀ a b, degt a + degt b ≤ degt (a + b)) (s : Finset ι) (f : ι → R[A]) :
    (∑ i ∈ s, (f i).support.inf degt) ≤ (∏ i ∈ s, f i).support.inf degt :=
                     /-
                       R : Type u_1
                       A : Type u_3
                       T : Type u_4
                       ι : Type u_6
                       inst✝⁶ : SemilatticeInf T
                       inst✝⁵ : OrderTop T
                       inst✝⁴ : CommSemiring R
                       inst✝³ : AddCommMonoid A
                       inst✝² : AddCommMonoid T
                       inst✝¹ : AddLeftMono T
                       inst✝ : AddRightMono T
                       degt : A → T
                       degt0 : LE.le 0 (degt 0)
                       degtm : ∀ (a b : A), LE.le (HAdd.hAdd (degt a) (degt b)) (degt (HAdd.hAdd a b))
                       s : Finset ι
                       f : ι → AddMonoidAlgebra R A
                       ⊢ Eq (s.sum fun i => (f i).support.inf degt) (Multiset.map (fun f => f.support …
                     -/
  le_of_eq_of_le (by rw [Multiset.map_map]; rfl) (le_inf_support_multiset_prod degt0 degtm _)
                                            /-
                                              🎉 no goals
                                            -/


/-- Let `R` be a semiring, let `A` be an `AddZeroClass`, let `B` be an `OrderBot`,
and let `D : A → B` be a "degree" function.
For an element `f : R[A]`, the element `supDegree f : B` is the supremum of all the elements in the
support of `f`, or `⊥` if `f` is zero.
Often, the Type `B` is `WithBot A`,
If, further, `A` has a linear order, then this notion coincides with the usual one,
using the maximum of the exponents.

If `A := σ →₀ ℕ` then `R[A] = MvPolynomial σ R`, and if we equip `σ` with a linear order then
the induced linear order on `Lex A` equips `MvPolynomial` ring with a
[monomial order](https://en.wikipedia.org/wiki/Monomial_order) (i.e. a linear order on `A`, the
type of (monic) monomials in `R[A]`, that respects addition). We make use of this monomial order
by taking `D := toLex`, and different monomial orders could be accessed via different type
synonyms once they are added. -/
abbrev supDegree (f : R[A]) : B :=
  f.support.sup D


theorem supDegree_add_le {f g : R[A]} :
    (f + g).supDegree D ≤ (f.supDegree D) ⊔ (g.supDegree D) :=
  sup_support_add_le D f g


@[simp]
theorem supDegree_neg {f : R'[A]} :
    (-f).supDegree D = f.supDegree D := by
  /-
    R' : Type u_2
    A : Type u_3
    B : Type u_5
    inst✝² : Ring R'
    inst✝¹ : SemilatticeSup B
    inst✝ : OrderBot B
    D : A → B
    f : AddMonoidAlgebra R' A
    ⊢ Eq (AddMonoidAlgebra.supDegree D (Neg.neg f)) (AddMonoidAlgebra.supDegree D f)
  -/
  rw [supDegree, supDegree, Finsupp.support_neg]
  /-
    🎉 no goals
  -/


theorem supDegree_sub_le {f g : R'[A]} :
    (f - g).supDegree D ≤ f.supDegree D ⊔ g.supDegree D := by
  /-
    R' : Type u_2
    A : Type u_3
    B : Type u_5
    inst✝² : Ring R'
    inst✝¹ : SemilatticeSup B
    inst✝ : OrderBot B
    D : A → B
    f g : AddMonoidAlgebra R' A
    ⊢ LE.le (AddMonoidAlgebra.supDegree D (HSub.hSub f g)) (Max.max (AddMonoidAlge …
  -/
  rw [sub_eq_add_neg, ← supDegree_neg (f := g)]; apply supDegree_add_le
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem supDegree_sum_le {ι} {s : Finset ι} {f : ι → R[A]} :
    (∑ i ∈ s, f i).supDegree D ≤ s.sup (fun i => (f i).supDegree D) := by
  classical
  exact (Finset.sup_mono Finsupp.support_finset_sum).trans_eq (Finset.sup_biUnion _ _)


theorem supDegree_single_ne_zero (a : A) {r : R} (hr : r ≠ 0) :
    (single a r).supDegree D = D a := by
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝² : Semiring R
    inst✝¹ : SemilatticeSup B
    inst✝ : OrderBot B
    D : A → B
    a : A
    r : R
    hr : Ne r 0
    ⊢ Eq (AddMonoidAlgebra.supDegree D (AddMonoidAlgebra.single a r)) (D a)
  -/
  rw [supDegree, Finsupp.support_single_ne_zero a hr, Finset.sup_singleton]
  /-
    🎉 no goals
  -/


open Classical in
theorem supDegree_single (a : A) (r : R) :
    (single a r).supDegree D = if r = 0 then ⊥ else D a := by
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝² : Semiring R
    inst✝¹ : SemilatticeSup B
    inst✝ : OrderBot B
    D : A → B
    a : A
    r : R
    ⊢ Eq (AddMonoidAlgebra.supDegree D (AddMonoidAlgebra.single a r)) (ite (Eq r 0 …
  -/
                        /-
                          🎉 no goals
                        -/
  split_ifs with hr <;> simp [supDegree_single_ne_zero, hr]
                        /-
                          🎉 no goals
                        -/


theorem apply_eq_zero_of_not_le_supDegree {p : R[A]} {a : A} (hlt : ¬ D a ≤ p.supDegree D) :
    p a = 0 := by
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝² : Semiring R
    inst✝¹ : SemilatticeSup B
    inst✝ : OrderBot B
    D : A → B
    p : AddMonoidAlgebra R A
    a : A
    hlt : Not (LE.le (D a) (AddMonoidAlgebra.supDegree D p))
    ⊢ Eq (p a) 0
  -/
  contrapose! hlt
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝² : Semiring R
    inst✝¹ : SemilatticeSup B
    inst✝ : OrderBot B
    D : A → B
    p : AddMonoidAlgebra R A
    a : A
    hlt : Ne (p a) 0
    ⊢ LE.le (D a) (AddMonoidAlgebra.supDegree D p)
  -/
  exact Finset.le_sup (Finsupp.mem_support_iff.2 hlt)
  /-
    🎉 no goals
  -/


theorem supDegree_withBot_some_comp {s : AddMonoidAlgebra R A} (hs : s.support.Nonempty) :
    supDegree (WithBot.some ∘ D) s = supDegree D s := by
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝² : Semiring R
    inst✝¹ : SemilatticeSup B
    inst✝ : OrderBot B
    D : A → B
    s : AddMonoidAlgebra R A
    hs : s.support.Nonempty
    ⊢ Eq (AddMonoidAlgebra.supDegree (Function.comp WithBot.some D) s) ↑(AddMonoid …
  -/
  unfold AddMonoidAlgebra.supDegree
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝² : Semiring R
    inst✝¹ : SemilatticeSup B
    inst✝ : OrderBot B
    D : A → B
    s : AddMonoidAlgebra R A
    hs : s.support.Nonempty
    ⊢ Eq (s.support.sup (Function.comp WithBot.some D)) ↑(s.support.sup D)
  -/
  rw [← Finset.coe_sup' hs, Finset.sup'_eq_sup]
  /-
    🎉 no goals
  -/


theorem supDegree_eq_of_isMaxOn {p : R[A]} {a : A} (hmem : a ∈ p.support)
    (hmax : IsMaxOn D p.support a) : p.supDegree D = D a :=
  sup_eq_of_isMaxOn hmem hmax


@[simp]
                                                          /-
                                                            R : Type u_1
                                                            A : Type u_3
                                                            B : Type u_5
                                                            inst✝³ : Semiring R
                                                            inst✝² : SemilatticeSup B
                                                            inst✝¹ : OrderBot B
                                                            D : A → B
                                                            inst✝ : AddZeroClass A
                                                            ⊢ Eq (AddMonoidAlgebra.supDegree D 0) Bot.bot
                                                          -/
theorem supDegree_zero : (0 : R[A]).supDegree D = ⊥ := by simp [supDegree]
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem ne_zero_of_supDegree_ne_bot : p.supDegree D ≠ ⊥ → p ≠ 0 := mt (fun h => h ▸ supDegree_zero)


theorem ne_zero_of_not_supDegree_le {b : B} (h : ¬ p.supDegree D ≤ b) : p ≠ 0 :=
  ne_zero_of_supDegree_ne_bot (fun he => h <| he ▸ bot_le)


theorem supDegree_eq_of_max {b : B} (hb : b ∈ Set.range D) (hmem : D.invFun b ∈ p.support)
    (hmax : ∀ a ∈ p.support, D a ≤ b) : p.supDegree D = b :=
  sup_eq_of_max hb hmem hmax


theorem supDegree_mul_le (hadd : ∀ a1 a2, D (a1 + a2) = D a1 + D a2)
    [AddLeftMono B] [AddRightMono B] :
    (p * q).supDegree D ≤ p.supDegree D + q.supDegree D :=
  sup_support_mul_le (fun {_ _} => (hadd _ _).le) p q


theorem supDegree_prod_le {R A B : Type*} [CommSemiring R] [AddCommMonoid A] [AddCommMonoid B]
    [SemilatticeSup B] [OrderBot B]
    [AddLeftMono B] [AddRightMono B]
    {D : A → B} (hzero : D 0 = 0) (hadd : ∀ a1 a2, D (a1 + a2) = D a1 + D a2)
    {ι} {s : Finset ι} {f : ι → R[A]} :
    (∏ i ∈ s, f i).supDegree D ≤ ∑ i ∈ s, (f i).supDegree D := by
  classical
  refine s.induction ?_ ?_
  · rw [Finset.prod_empty, Finset.sum_empty, one_def, supDegree_single]
    split_ifs; exacts [bot_le, hzero.le]
  · intro i s his ih
    rw [Finset.prod_insert his, Finset.sum_insert his]
    exact (supDegree_mul_le hadd).trans (by gcongr)


theorem apply_add_of_supDegree_le (hadd : ∀ a1 a2, D (a1 + a2) = D a1 + D a2)
    [AddLeftStrictMono B] [AddRightStrictMono B]
    (hD : D.Injective) {ap aq : A} (hp : p.supDegree D ≤ D ap) (hq : q.supDegree D ≤ D aq) :
    (p * q) (ap + aq) = p ap * q aq := by
  classical
  simp_rw [mul_apply, Finsupp.sum]
  rw [Finset.sum_eq_single ap, Finset.sum_eq_single aq, if_pos rfl]
  · refine fun a ha hne => if_neg (fun he => ?_)
    apply_fun D at he; simp_rw [hadd] at he
    exact (add_lt_add_left (((Finset.le_sup ha).trans hq).lt_of_ne <| hD.ne_iff.2 hne) _).ne he
  · intro h; rw [if_pos rfl, Finsupp.not_mem_support_iff.1 h, mul_zero]
  · refine fun a ha hne => Finset.sum_eq_zero (fun a' ha' => if_neg <| fun he => ?_)
    apply_fun D at he
    simp_rw [hadd] at he
    have := addLeftMono_of_addLeftStrictMono B
    exact (add_lt_add_of_lt_of_le (((Finset.le_sup ha).trans hp).lt_of_ne <| hD.ne_iff.2 hne)
      <| (Finset.le_sup ha').trans hq).ne he
  · refine fun h => Finset.sum_eq_zero (fun a _ => ite_eq_right_iff.mpr <| fun _ => ?_)
    rw [Finsupp.not_mem_support_iff.mp h, zero_mul]


/-- If `D` is an injection into a linear order `B`, the leading coefficient of `f : R[A]` is the
  nonzero coefficient of highest degree according to `D`, or 0 if `f = 0`. In general, it is defined
  to be the coefficient at an inverse image of `supDegree f` (if such exists). -/
noncomputable def leadingCoeff [Nonempty A] (f : R[A]) : R :=
  f (D.invFun <| f.supDegree D)


/-- An element `f : R[A]` is monic if its leading coefficient is one. -/
@[reducible] def Monic [Nonempty A] (f : R[A]) : Prop :=
  f.leadingCoeff D = 1


@[simp]
theorem leadingCoeff_single [Nonempty A] (hD : D.Injective) (a : A) (r : R) :
    (single a r).leadingCoeff D = r := by
  classical
  rw [leadingCoeff, supDegree_single]
  split_ifs with hr
  · simp [hr]
  · rw [Function.leftInverse_invFun hD, single_apply, if_pos rfl]


@[simp]
theorem leadingCoeff_zero [Nonempty A] : (0 : R[A]).leadingCoeff D = 0 := rfl


lemma Monic.ne_zero [Nonempty A] [Nontrivial R] (hp : p.Monic D) : p ≠ 0 := fun h => by
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁴ : Semiring R
    inst✝³ : LinearOrder B
    inst✝² : OrderBot B
    p : AddMonoidAlgebra R A
    D : A → B
    inst✝¹ : Nonempty A
    inst✝ : Nontrivial R
    hp : AddMonoidAlgebra.Monic D p
    h : Eq p 0
    ⊢ False
  -/
  simp_rw [Monic, h, leadingCoeff_zero, zero_ne_one] at hp
  /-
    🎉 no goals
  -/


@[simp]
theorem monic_one [AddZeroClass A] (hD : D.Injective) : (1 : R[A]).Monic D := by
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝³ : Semiring R
    inst✝² : LinearOrder B
    inst✝¹ : OrderBot B
    D : A → B
    inst✝ : AddZeroClass A
    hD : Function.Injective D
    ⊢ AddMonoidAlgebra.Monic D 1
  -/
  rw [Monic, one_def, leadingCoeff_single hD]
  /-
    🎉 no goals
  -/


variable (D) in
lemma exists_supDegree_mem_support (hp : p ≠ 0) : ∃ a ∈ p.support, p.supDegree D = D a :=
  Finset.exists_mem_eq_sup _ (Finsupp.support_nonempty_iff.mpr hp) D


variable (D) in
lemma supDegree_mem_range (hp : p ≠ 0) : p.supDegree D ∈ Set.range D := by
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝² : Semiring R
    inst✝¹ : LinearOrder B
    inst✝ : OrderBot B
    p : AddMonoidAlgebra R A
    D : A → B
    hp : Ne p 0
    ⊢ Membership.mem (Set.range D) (AddMonoidAlgebra.supDegree D p)
  -/
  obtain ⟨a, -, he⟩ := exists_supDegree_mem_support D hp; exact ⟨a, he.symm⟩
                                                          /-
                                                            🎉 no goals
                                                          -/


lemma supDegree_sum_lt (hs : s.Nonempty) {b : B}
    (h : ∀ i ∈ s, (f i).supDegree D < b) : (∑ i ∈ s, f i).supDegree D < b := by
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝² : Semiring R
    inst✝¹ : LinearOrder B
    inst✝ : OrderBot B
    D : A → B
    ι : Type u_7
    s : Finset ι
    f : ι → AddMonoidAlgebra R A
    hs : s.Nonempty
    b : B
    h : ∀ (i : ι), Membership.mem s i → LT.lt (AddMonoidAlgebra.supDegree D (f i)) b
    ⊢ LT.lt (AddMonoidAlgebra.supDegree D (s.sum fun i => f i)) b
  -/
  refine supDegree_sum_le.trans_lt ((Finset.sup_lt_iff ?_).mpr h)
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝² : Semiring R
    inst✝¹ : LinearOrder B
    inst✝ : OrderBot B
    D : A → B
    ι : Type u_7
    s : Finset ι
    f : ι → AddMonoidAlgebra R A
    hs : s.Nonempty
    b : B
    h : ∀ (i : ι), Membership.mem s i → LT.lt (AddMonoidAlgebra.supDegree D (f i)) b
    ⊢ LT.lt Bot.bot b
  -/
  obtain ⟨i, hi⟩ := hs; exact bot_le.trans_lt (h i hi)
                        /-
                          🎉 no goals
                        -/


open Finsupp in
lemma supDegree_add_eq_left (h : q.supDegree D < p.supDegree D) :
    (p + q).supDegree D = p.supDegree D := by
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝³ : Semiring R
    inst✝² : LinearOrder B
    inst✝¹ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝ : AddZeroClass A
    h : LT.lt (AddMonoidAlgebra.supDegree D q) (AddMonoidAlgebra.supDegree D p)
    ⊢ Eq (AddMonoidAlgebra.supDegree D (HAdd.hAdd p q)) (AddMonoidAlgebra.supDegre …
  -/
  apply (supDegree_add_le.trans <| sup_le le_rfl h.le).antisymm
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝³ : Semiring R
    inst✝² : LinearOrder B
    inst✝¹ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝ : AddZeroClass A
    h : LT.lt (AddMonoidAlgebra.supDegree D q) (AddMonoidAlgebra.supDegree D p)
    ⊢ LE.le (AddMonoidAlgebra.supDegree D p) (AddMonoidAlgebra.supDegree D (HAdd.h …
  -/
  obtain ⟨a, ha, he⟩ := exists_supDegree_mem_support D (ne_zero_of_not_supDegree_le h.not_le)
  /-
    case intro.intro
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝³ : Semiring R
    inst✝² : LinearOrder B
    inst✝¹ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝ : AddZeroClass A
    h : LT.lt (AddMonoidAlgebra.supDegree D q) (AddMonoidAlgebra.supDegree D p)
    a : A
    ha : Membership.mem p.support a
    he : Eq (AddMonoidAlgebra.supDegree D p) (D a)
    ⊢ LE.le (AddMonoidAlgebra.supDegree D p) (AddMonoidAlgebra.supDegree D (HAdd.h …
  -/
  rw [he] at h ⊢
  /-
    case intro.intro
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝³ : Semiring R
    inst✝² : LinearOrder B
    inst✝¹ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝ : AddZeroClass A
    a : A
    h : LT.lt (AddMonoidAlgebra.supDegree D q) (D a)
    ha : Membership.mem p.support a
    he : Eq (AddMonoidAlgebra.supDegree D p) (D a)
    ⊢ LE.le (D a) (AddMonoidAlgebra.supDegree D (HAdd.hAdd p q))
  -/
  apply Finset.le_sup
  /-
    case intro.intro.hb
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝³ : Semiring R
    inst✝² : LinearOrder B
    inst✝¹ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝ : AddZeroClass A
    a : A
    h : LT.lt (AddMonoidAlgebra.supDegree D q) (D a)
    ha : Membership.mem p.support a
    he : Eq (AddMonoidAlgebra.supDegree D p) (D a)
    ⊢ Membership.mem (HAdd.hAdd p q).support a
  -/
  rw [mem_support_iff, add_apply, apply_eq_zero_of_not_le_supDegree h.not_le, add_zero]
  /-
    case intro.intro.hb
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝³ : Semiring R
    inst✝² : LinearOrder B
    inst✝¹ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝ : AddZeroClass A
    a : A
    h : LT.lt (AddMonoidAlgebra.supDegree D q) (D a)
    ha : Membership.mem p.support a
    he : Eq (AddMonoidAlgebra.supDegree D p) (D a)
    ⊢ Ne (p a) 0
  -/
  exact mem_support_iff.mp ha
  /-
    🎉 no goals
  -/


lemma supDegree_add_eq_right (h : p.supDegree D < q.supDegree D) :
    (p + q).supDegree D = q.supDegree D := by
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝³ : Semiring R
    inst✝² : LinearOrder B
    inst✝¹ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝ : AddZeroClass A
    h : LT.lt (AddMonoidAlgebra.supDegree D p) (AddMonoidAlgebra.supDegree D q)
    ⊢ Eq (AddMonoidAlgebra.supDegree D (HAdd.hAdd p q)) (AddMonoidAlgebra.supDegre …
  -/
  rw [add_comm, supDegree_add_eq_left h]
  /-
    🎉 no goals
  -/


lemma leadingCoeff_add_eq_left (h : q.supDegree D < p.supDegree D) :
    (p + q).leadingCoeff D = p.leadingCoeff D := by
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝³ : Semiring R
    inst✝² : LinearOrder B
    inst✝¹ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝ : AddZeroClass A
    h : LT.lt (AddMonoidAlgebra.supDegree D q) (AddMonoidAlgebra.supDegree D p)
    ⊢ Eq (AddMonoidAlgebra.leadingCoeff D (HAdd.hAdd p q)) (AddMonoidAlgebra.leadi …
  -/
  obtain ⟨a, he⟩ := supDegree_mem_range D (ne_zero_of_not_supDegree_le h.not_le)
  rw [leadingCoeff, supDegree_add_eq_left h, Finsupp.add_apply, ← leadingCoeff,
    apply_eq_zero_of_not_le_supDegree (D := D), add_zero]
  /-
    case intro
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝³ : Semiring R
    inst✝² : LinearOrder B
    inst✝¹ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝ : AddZeroClass A
    h : LT.lt (AddMonoidAlgebra.supDegree D q) (AddMonoidAlgebra.supDegree D p)
    a : A
    he : Eq (D a) (AddMonoidAlgebra.supDegree D p)
    ⊢ Not (LE.le (D (Function.invFun D (AddMonoidAlgebra.supDegree D p))) (AddMono …
  -/
  rw [← he, Function.apply_invFun_apply (f := D), he]; exact h.not_le
                                                       /-
                                                         🎉 no goals
                                                       -/


lemma leadingCoeff_add_eq_right (h : p.supDegree D < q.supDegree D) :
    (p + q).leadingCoeff D = q.leadingCoeff D := by
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝³ : Semiring R
    inst✝² : LinearOrder B
    inst✝¹ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝ : AddZeroClass A
    h : LT.lt (AddMonoidAlgebra.supDegree D p) (AddMonoidAlgebra.supDegree D q)
    ⊢ Eq (AddMonoidAlgebra.leadingCoeff D (HAdd.hAdd p q)) (AddMonoidAlgebra.leadi …
  -/
  rw [add_comm, leadingCoeff_add_eq_left h]
  /-
    🎉 no goals
  -/


lemma supDegree_mem_support (hD : D.Injective) (hp : p ≠ 0) :
    D.invFun (p.supDegree D) ∈ p.support := by
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝³ : Semiring R
    inst✝² : LinearOrder B
    inst✝¹ : OrderBot B
    p : AddMonoidAlgebra R A
    D : A → B
    inst✝ : AddZeroClass A
    hD : Function.Injective D
    hp : Ne p 0
    ⊢ Membership.mem p.support (Function.invFun D (AddMonoidAlgebra.supDegree D p))
  -/
  obtain ⟨a, ha, he⟩ := exists_supDegree_mem_support D hp
  /-
    case intro.intro
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝³ : Semiring R
    inst✝² : LinearOrder B
    inst✝¹ : OrderBot B
    p : AddMonoidAlgebra R A
    D : A → B
    inst✝ : AddZeroClass A
    hD : Function.Injective D
    hp : Ne p 0
    a : A
    ha : Membership.mem p.support a
    he : Eq (AddMonoidAlgebra.supDegree D p) (D a)
    ⊢ Membership.mem p.support (Function.invFun D (AddMonoidAlgebra.supDegree D p))
  -/
  rwa [he, Function.leftInverse_invFun hD]
  /-
    🎉 no goals
  -/


@[simp]
lemma leadingCoeff_eq_zero (hD : D.Injective) : p.leadingCoeff D = 0 ↔ p = 0 := by
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝³ : Semiring R
    inst✝² : LinearOrder B
    inst✝¹ : OrderBot B
    p : AddMonoidAlgebra R A
    D : A → B
    inst✝ : AddZeroClass A
    hD : Function.Injective D
    ⊢ Iff (Eq (AddMonoidAlgebra.leadingCoeff D p) 0) (Eq p 0)
  -/
  refine ⟨(fun h => ?_).mtr, fun h => h ▸ leadingCoeff_zero⟩
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝³ : Semiring R
    inst✝² : LinearOrder B
    inst✝¹ : OrderBot B
    p : AddMonoidAlgebra R A
    D : A → B
    inst✝ : AddZeroClass A
    hD : Function.Injective D
    h : Not (Eq p 0)
    ⊢ Not (Eq (AddMonoidAlgebra.leadingCoeff D p) 0)
  -/
  rw [leadingCoeff, ← Ne, ← Finsupp.mem_support_iff]
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝³ : Semiring R
    inst✝² : LinearOrder B
    inst✝¹ : OrderBot B
    p : AddMonoidAlgebra R A
    D : A → B
    inst✝ : AddZeroClass A
    hD : Function.Injective D
    h : Not (Eq p 0)
    ⊢ Membership.mem p.support (Function.invFun D (AddMonoidAlgebra.supDegree D p))
  -/
  exact supDegree_mem_support hD h
  /-
    🎉 no goals
  -/


lemma leadingCoeff_ne_zero (hD : D.Injective) : p.leadingCoeff D ≠ 0 ↔ p ≠ 0 :=
  (leadingCoeff_eq_zero hD).ne


lemma supDegree_sub_lt_of_leadingCoeff_eq (hD : D.Injective) {R} [CommRing R] {p q : R[A]}
    (hd : p.supDegree D = q.supDegree D) (hc : p.leadingCoeff D = q.leadingCoeff D) :
    (p - q).supDegree D < p.supDegree D ∨ p = q := by
  /-
    A : Type u_3
    B : Type u_5
    inst✝³ : LinearOrder B
    inst✝² : OrderBot B
    D : A → B
    inst✝¹ : AddZeroClass A
    hD : Function.Injective D
    R : Type u_8
    inst✝ : CommRing R
    p q : AddMonoidAlgebra R A
    hd : Eq (AddMonoidAlgebra.supDegree D p) (AddMonoidAlgebra.supDegree D q)
    hc : Eq (AddMonoidAlgebra.leadingCoeff D p) (AddMonoidAlgebra.leadingCoeff D q)
    ⊢ Or (LT.lt (AddMonoidAlgebra.supDegree D (HSub.hSub p q)) (AddMonoidAlgebra.s …
  -/
  rw [or_iff_not_imp_right]
  /-
    A : Type u_3
    B : Type u_5
    inst✝³ : LinearOrder B
    inst✝² : OrderBot B
    D : A → B
    inst✝¹ : AddZeroClass A
    hD : Function.Injective D
    R : Type u_8
    inst✝ : CommRing R
    p q : AddMonoidAlgebra R A
    hd : Eq (AddMonoidAlgebra.supDegree D p) (AddMonoidAlgebra.supDegree D q)
    hc : Eq (AddMonoidAlgebra.leadingCoeff D p) (AddMonoidAlgebra.leadingCoeff D q)
    ⊢ Not (Eq p q) → LT.lt (AddMonoidAlgebra.supDegree D (HSub.hSub p q)) (AddMono …
  -/
  refine fun he => (supDegree_sub_le.trans ?_).lt_of_ne ?_
    /-
      case refine_1
      A : Type u_3
      B : Type u_5
      inst✝³ : LinearOrder B
      inst✝² : OrderBot B
      D : A → B
      inst✝¹ : AddZeroClass A
      hD : Function.Injective D
      R : Type u_8
      inst✝ : CommRing R
      p q : AddMonoidAlgebra R A
      hd : Eq (AddMonoidAlgebra.supDegree D p) (AddMonoidAlgebra.supDegree D q)
      hc : Eq (AddMonoidAlgebra.leadingCoeff D p) (AddMonoidAlgebra.leadingCoeff D q)
      he : Not (Eq p q)
      ⊢ LE.le (Max.max (AddMonoidAlgebra.supDegree D p) (AddMonoidAlgebra.supDegree  …
    -/
  · rw [hd, sup_idem]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      A : Type u_3
      B : Type u_5
      inst✝³ : LinearOrder B
      inst✝² : OrderBot B
      D : A → B
      inst✝¹ : AddZeroClass A
      hD : Function.Injective D
      R : Type u_8
      inst✝ : CommRing R
      p q : AddMonoidAlgebra R A
      hd : Eq (AddMonoidAlgebra.supDegree D p) (AddMonoidAlgebra.supDegree D q)
      hc : Eq (AddMonoidAlgebra.leadingCoeff D p) (AddMonoidAlgebra.leadingCoeff D q)
      he : Not (Eq p q)
      ⊢ Ne (AddMonoidAlgebra.supDegree D (HSub.hSub p q)) (AddMonoidAlgebra.supDegre …
    -/
  · rw [← sub_eq_zero, ← leadingCoeff_eq_zero hD, leadingCoeff] at he
    /-
      case refine_2
      A : Type u_3
      B : Type u_5
      inst✝³ : LinearOrder B
      inst✝² : OrderBot B
      D : A → B
      inst✝¹ : AddZeroClass A
      hD : Function.Injective D
      R : Type u_8
      inst✝ : CommRing R
      p q : AddMonoidAlgebra R A
      hd : Eq (AddMonoidAlgebra.supDegree D p) (AddMonoidAlgebra.supDegree D q)
      hc : Eq (AddMonoidAlgebra.leadingCoeff D p) (AddMonoidAlgebra.leadingCoeff D q)
      he : Not (Eq ((HSub.hSub p q) (Function.invFun D (AddMonoidAlgebra.supDegree D …
      ⊢ Ne (AddMonoidAlgebra.supDegree D (HSub.hSub p q)) (AddMonoidAlgebra.supDegre …
    -/
    refine fun h => he ?_
    /-
      case refine_2
      A : Type u_3
      B : Type u_5
      inst✝³ : LinearOrder B
      inst✝² : OrderBot B
      D : A → B
      inst✝¹ : AddZeroClass A
      hD : Function.Injective D
      R : Type u_8
      inst✝ : CommRing R
      p q : AddMonoidAlgebra R A
      hd : Eq (AddMonoidAlgebra.supDegree D p) (AddMonoidAlgebra.supDegree D q)
      hc : Eq (AddMonoidAlgebra.leadingCoeff D p) (AddMonoidAlgebra.leadingCoeff D q)
      he : Not (Eq ((HSub.hSub p q) (Function.invFun D (AddMonoidAlgebra.supDegree D …
      h : Eq (AddMonoidAlgebra.supDegree D (HSub.hSub p q)) (AddMonoidAlgebra.supDeg …
      ⊢ Eq ((HSub.hSub p q) (Function.invFun D (AddMonoidAlgebra.supDegree D (HSub.h …
    -/
    rwa [h, Finsupp.sub_apply, ← leadingCoeff, hd, ← leadingCoeff, sub_eq_zero]
    /-
      🎉 no goals
    -/


lemma supDegree_leadingCoeff_sum_eq
    (hi : i ∈ s) (hmax : ∀ j ∈ s, j ≠ i → (f j).supDegree D < (f i).supDegree D) :
    (∑ j ∈ s, f j).supDegree D = (f i).supDegree D ∧
    (∑ j ∈ s, f j).leadingCoeff D = (f i).leadingCoeff D := by
  classical
  rw [← s.add_sum_erase _ hi]
  by_cases hs : s.erase i = ∅
  · rw [hs, Finset.sum_empty, add_zero]; exact ⟨rfl, rfl⟩
  suffices _ from ⟨supDegree_add_eq_left this, leadingCoeff_add_eq_left this⟩
  refine supDegree_sum_lt ?_ (fun j hj => ?_)
  · rw [Finset.nonempty_iff_ne_empty]; exact hs
  · rw [Finset.mem_erase] at hj; exact hmax j hj.2 hj.1


open Finset in
lemma sum_ne_zero_of_injOn_supDegree' (hs : ∃ i ∈ s, f i ≠ 0)
    (hd : (s : Set ι).InjOn (supDegree D ∘ f)) :
    ∑ i ∈ s, f i ≠ 0 := by
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝³ : Semiring R
    inst✝² : LinearOrder B
    inst✝¹ : OrderBot B
    D : A → B
    ι : Type u_7
    s : Finset ι
    f : ι → AddMonoidAlgebra R A
    inst✝ : AddZeroClass A
    hs : Exists fun i => And (Membership.mem s i) (Ne (f i) 0)
    hd : Set.InjOn (Function.comp (AddMonoidAlgebra.supDegree D) f) ↑s
    ⊢ Ne (s.sum fun i => f i) 0
  -/
  obtain ⟨j, hj, hne⟩ := hs
  /-
    case intro.intro
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝³ : Semiring R
    inst✝² : LinearOrder B
    inst✝¹ : OrderBot B
    D : A → B
    ι : Type u_7
    s : Finset ι
    f : ι → AddMonoidAlgebra R A
    inst✝ : AddZeroClass A
    hd : Set.InjOn (Function.comp (AddMonoidAlgebra.supDegree D) f) ↑s
    j : ι
    hj : Membership.mem s j
    hne : Ne (f j) 0
    ⊢ Ne (s.sum fun i => f i) 0
  -/
  obtain ⟨i, hi, he⟩ := exists_mem_eq_sup _ ⟨j, hj⟩ (supDegree D ∘ f)
  /-
    case intro.intro.intro.intro
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝³ : Semiring R
    inst✝² : LinearOrder B
    inst✝¹ : OrderBot B
    D : A → B
    ι : Type u_7
    s : Finset ι
    f : ι → AddMonoidAlgebra R A
    inst✝ : AddZeroClass A
    hd : Set.InjOn (Function.comp (AddMonoidAlgebra.supDegree D) f) ↑s
    j : ι
    hj : Membership.mem s j
    hne : Ne (f j) 0
    i : ι
    hi : Membership.mem s i
    he : Eq (s.sup (Function.comp (AddMonoidAlgebra.supDegree D) f)) (Function.com …
    ⊢ Ne (s.sum fun i => f i) 0
  -/
  by_cases h : ∀ k ∈ s, k = i
    /-
      case pos
      R : Type u_1
      A : Type u_3
      B : Type u_5
      inst✝³ : Semiring R
      inst✝² : LinearOrder B
      inst✝¹ : OrderBot B
      D : A → B
      ι : Type u_7
      s : Finset ι
      f : ι → AddMonoidAlgebra R A
      inst✝ : AddZeroClass A
      hd : Set.InjOn (Function.comp (AddMonoidAlgebra.supDegree D) f) ↑s
      j : ι
      hj : Membership.mem s j
      hne : Ne (f j) 0
      i : ι
      hi : Membership.mem s i
      he : Eq (s.sup (Function.comp (AddMonoidAlgebra.supDegree D) f)) (Function.com …
      h : ∀ (k : ι), Membership.mem s k → Eq k i
      ⊢ Ne (s.sum fun i => f i) 0
    -/
  · refine (sum_eq_single_of_mem j hj (fun k hk hne => ?_)).trans_ne hne
    /-
      case pos
      R : Type u_1
      A : Type u_3
      B : Type u_5
      inst✝³ : Semiring R
      inst✝² : LinearOrder B
      inst✝¹ : OrderBot B
      D : A → B
      ι : Type u_7
      s : Finset ι
      f : ι → AddMonoidAlgebra R A
      inst✝ : AddZeroClass A
      hd : Set.InjOn (Function.comp (AddMonoidAlgebra.supDegree D) f) ↑s
      j : ι
      hj : Membership.mem s j
      hne✝ : Ne (f j) 0
      i : ι
      hi : Membership.mem s i
      he : Eq (s.sup (Function.comp (AddMonoidAlgebra.supDegree D) f)) (Function.com …
      h : ∀ (k : ι), Membership.mem s k → Eq k i
      k : ι
      hk : Membership.mem s k
      hne : Ne k j
      ⊢ Eq (f k) 0
    -/
    rw [h k hk, h j hj] at hne; exact hne.irrefl.elim
                                /-
                                  🎉 no goals
                                -/
  /-
    case neg
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝³ : Semiring R
    inst✝² : LinearOrder B
    inst✝¹ : OrderBot B
    D : A → B
    ι : Type u_7
    s : Finset ι
    f : ι → AddMonoidAlgebra R A
    inst✝ : AddZeroClass A
    hd : Set.InjOn (Function.comp (AddMonoidAlgebra.supDegree D) f) ↑s
    j : ι
    hj : Membership.mem s j
    hne : Ne (f j) 0
    i : ι
    hi : Membership.mem s i
    he : Eq (s.sup (Function.comp (AddMonoidAlgebra.supDegree D) f)) (Function.com …
    h : Not (∀ (k : ι), Membership.mem s k → Eq k i)
    ⊢ Ne (s.sum fun i => f i) 0
  -/
  push_neg at h; obtain ⟨j, hj, hne⟩ := h
  /-
    case neg.intro.intro
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝³ : Semiring R
    inst✝² : LinearOrder B
    inst✝¹ : OrderBot B
    D : A → B
    ι : Type u_7
    s : Finset ι
    f : ι → AddMonoidAlgebra R A
    inst✝ : AddZeroClass A
    hd : Set.InjOn (Function.comp (AddMonoidAlgebra.supDegree D) f) ↑s
    j✝ : ι
    hj✝ : Membership.mem s j✝
    hne✝ : Ne (f j✝) 0
    i : ι
    hi : Membership.mem s i
    he : Eq (s.sup (Function.comp (AddMonoidAlgebra.supDegree D) f)) (Function.com …
    j : ι
    hj : Membership.mem s j
    hne : Ne j i
    ⊢ Ne (s.sum fun i => f i) 0
  -/
  apply ne_zero_of_supDegree_ne_bot (D := D)
  have (k) (hk : k ∈ s) (hne : k ≠ i) : supDegree D (f k) < supDegree D (f i) :=
    ((le_sup hk).trans_eq he).lt_of_ne (hd.ne hk hi hne)
  /-
    case neg.intro.intro
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝³ : Semiring R
    inst✝² : LinearOrder B
    inst✝¹ : OrderBot B
    D : A → B
    ι : Type u_7
    s : Finset ι
    f : ι → AddMonoidAlgebra R A
    inst✝ : AddZeroClass A
    hd : Set.InjOn (Function.comp (AddMonoidAlgebra.supDegree D) f) ↑s
    j✝ : ι
    hj✝ : Membership.mem s j✝
    hne✝ : Ne (f j✝) 0
    i : ι
    hi : Membership.mem s i
    he : Eq (s.sup (Function.comp (AddMonoidAlgebra.supDegree D) f)) (Function.com …
    j : ι
    hj : Membership.mem s j
    hne : Ne j i
    this : ∀ (k : ι), Membership.mem s k → Ne k i → LT.lt (AddMonoidAlgebra.supDeg …
    ⊢ Ne (AddMonoidAlgebra.supDegree D (s.sum fun i => f i)) Bot.bot
  -/
  rw [(supDegree_leadingCoeff_sum_eq hi this).1]
  /-
    case neg.intro.intro
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝³ : Semiring R
    inst✝² : LinearOrder B
    inst✝¹ : OrderBot B
    D : A → B
    ι : Type u_7
    s : Finset ι
    f : ι → AddMonoidAlgebra R A
    inst✝ : AddZeroClass A
    hd : Set.InjOn (Function.comp (AddMonoidAlgebra.supDegree D) f) ↑s
    j✝ : ι
    hj✝ : Membership.mem s j✝
    hne✝ : Ne (f j✝) 0
    i : ι
    hi : Membership.mem s i
    he : Eq (s.sup (Function.comp (AddMonoidAlgebra.supDegree D) f)) (Function.com …
    j : ι
    hj : Membership.mem s j
    hne : Ne j i
    this : ∀ (k : ι), Membership.mem s k → Ne k i → LT.lt (AddMonoidAlgebra.supDeg …
    ⊢ Ne (AddMonoidAlgebra.supDegree D (f i)) Bot.bot
  -/
  exact (this j hj hne).ne_bot
  /-
    🎉 no goals
  -/


lemma sum_ne_zero_of_injOn_supDegree (hs : s ≠ ∅)
    (hf : ∀ i ∈ s, f i ≠ 0) (hd : (s : Set ι).InjOn (supDegree D ∘ f)) :
    ∑ i ∈ s, f i ≠ 0 :=
  let ⟨i, hi⟩ := Finset.nonempty_iff_ne_empty.2 hs
  sum_ne_zero_of_injOn_supDegree' ⟨i, hi, hf i hi⟩ hd


lemma apply_supDegree_add_supDegree (hD : D.Injective) (hadd : ∀ a1 a2, D (a1 + a2) = D a1 + D a2) :
    (p * q) (D.invFun (p.supDegree D + q.supDegree D)) = p.leadingCoeff D * q.leadingCoeff D := by
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : LinearOrder B
    inst✝⁴ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝³ : AddZeroClass A
    inst✝² : Add B
    inst✝¹ : AddLeftStrictMono B
    inst✝ : AddRightStrictMono B
    hD : Function.Injective D
    hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
    ⊢ Eq ((HMul.hMul p q) (Function.invFun D (HAdd.hAdd (AddMonoidAlgebra.supDegre …
  -/
  obtain rfl | hp := eq_or_ne p 0
    /-
      case inl
      R : Type u_1
      A : Type u_3
      B : Type u_5
      inst✝⁶ : Semiring R
      inst✝⁵ : LinearOrder B
      inst✝⁴ : OrderBot B
      q : AddMonoidAlgebra R A
      D : A → B
      inst✝³ : AddZeroClass A
      inst✝² : Add B
      inst✝¹ : AddLeftStrictMono B
      inst✝ : AddRightStrictMono B
      hD : Function.Injective D
      hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
      ⊢ Eq ((HMul.hMul 0 q) (Function.invFun D (HAdd.hAdd (AddMonoidAlgebra.supDegre …
    -/
  · simp_rw [leadingCoeff_zero, zero_mul, Finsupp.coe_zero, Pi.zero_apply]
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : LinearOrder B
    inst✝⁴ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝³ : AddZeroClass A
    inst✝² : Add B
    inst✝¹ : AddLeftStrictMono B
    inst✝ : AddRightStrictMono B
    hD : Function.Injective D
    hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
    hp : Ne p 0
    ⊢ Eq ((HMul.hMul p q) (Function.invFun D (HAdd.hAdd (AddMonoidAlgebra.supDegre …
  -/
  obtain rfl | hq := eq_or_ne q 0
    /-
      case inr.inl
      R : Type u_1
      A : Type u_3
      B : Type u_5
      inst✝⁶ : Semiring R
      inst✝⁵ : LinearOrder B
      inst✝⁴ : OrderBot B
      p : AddMonoidAlgebra R A
      D : A → B
      inst✝³ : AddZeroClass A
      inst✝² : Add B
      inst✝¹ : AddLeftStrictMono B
      inst✝ : AddRightStrictMono B
      hD : Function.Injective D
      hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
      hp : Ne p 0
      ⊢ Eq ((HMul.hMul p 0) (Function.invFun D (HAdd.hAdd (AddMonoidAlgebra.supDegre …
    -/
  · simp_rw [leadingCoeff_zero, mul_zero, Finsupp.coe_zero, Pi.zero_apply]
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : LinearOrder B
    inst✝⁴ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝³ : AddZeroClass A
    inst✝² : Add B
    inst✝¹ : AddLeftStrictMono B
    inst✝ : AddRightStrictMono B
    hD : Function.Injective D
    hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
    hp : Ne p 0
    hq : Ne q 0
    ⊢ Eq ((HMul.hMul p q) (Function.invFun D (HAdd.hAdd (AddMonoidAlgebra.supDegre …
  -/
  obtain ⟨ap, -, hp⟩ := exists_supDegree_mem_support D hp
  /-
    case inr.inr.intro.intro
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : LinearOrder B
    inst✝⁴ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝³ : AddZeroClass A
    inst✝² : Add B
    inst✝¹ : AddLeftStrictMono B
    inst✝ : AddRightStrictMono B
    hD : Function.Injective D
    hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
    hp✝ : Ne p 0
    hq : Ne q 0
    ap : A
    hp : Eq (AddMonoidAlgebra.supDegree D p) (D ap)
    ⊢ Eq ((HMul.hMul p q) (Function.invFun D (HAdd.hAdd (AddMonoidAlgebra.supDegre …
  -/
  obtain ⟨aq, -, hq⟩ := exists_supDegree_mem_support D hq
  /-
    case inr.inr.intro.intro.intro.intro
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : LinearOrder B
    inst✝⁴ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝³ : AddZeroClass A
    inst✝² : Add B
    inst✝¹ : AddLeftStrictMono B
    inst✝ : AddRightStrictMono B
    hD : Function.Injective D
    hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
    hp✝ : Ne p 0
    hq✝ : Ne q 0
    ap : A
    hp : Eq (AddMonoidAlgebra.supDegree D p) (D ap)
    aq : A
    hq : Eq (AddMonoidAlgebra.supDegree D q) (D aq)
    ⊢ Eq ((HMul.hMul p q) (Function.invFun D (HAdd.hAdd (AddMonoidAlgebra.supDegre …
  -/
  simp_rw [leadingCoeff, hp, hq, ← hadd, Function.leftInverse_invFun hD _]
  /-
    case inr.inr.intro.intro.intro.intro
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : LinearOrder B
    inst✝⁴ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝³ : AddZeroClass A
    inst✝² : Add B
    inst✝¹ : AddLeftStrictMono B
    inst✝ : AddRightStrictMono B
    hD : Function.Injective D
    hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
    hp✝ : Ne p 0
    hq✝ : Ne q 0
    ap : A
    hp : Eq (AddMonoidAlgebra.supDegree D p) (D ap)
    aq : A
    hq : Eq (AddMonoidAlgebra.supDegree D q) (D aq)
    ⊢ Eq ((HMul.hMul p q) (HAdd.hAdd ap aq)) (HMul.hMul (p ap) (q aq))
  -/
  exact apply_add_of_supDegree_le hadd hD hp.le hq.le
  /-
    🎉 no goals
  -/


lemma supDegree_mul
    (hD : D.Injective) (hadd : ∀ a1 a2, D (a1 + a2) = D a1 + D a2)
    (hpq : leadingCoeff D p * leadingCoeff D q ≠ 0)
    (hp : p ≠ 0) (hq : q ≠ 0) :
    (p * q).supDegree D = p.supDegree D + q.supDegree D := by
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : LinearOrder B
    inst✝⁴ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝³ : AddZeroClass A
    inst✝² : Add B
    inst✝¹ : AddLeftStrictMono B
    inst✝ : AddRightStrictMono B
    hD : Function.Injective D
    hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
    hpq : Ne (HMul.hMul (AddMonoidAlgebra.leadingCoeff D p) (AddMonoidAlgebra.lead …
    hp : Ne p 0
    hq : Ne q 0
    ⊢ Eq (AddMonoidAlgebra.supDegree D (HMul.hMul p q)) (HAdd.hAdd (AddMonoidAlgeb …
  -/
  cases subsingleton_or_nontrivial R; · exact (hp (Subsingleton.elim _ _)).elim
                                        /-
                                          🎉 no goals
                                        -/
  /-
    case inr
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : LinearOrder B
    inst✝⁴ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝³ : AddZeroClass A
    inst✝² : Add B
    inst✝¹ : AddLeftStrictMono B
    inst✝ : AddRightStrictMono B
    hD : Function.Injective D
    hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
    hpq : Ne (HMul.hMul (AddMonoidAlgebra.leadingCoeff D p) (AddMonoidAlgebra.lead …
    hp : Ne p 0
    hq : Ne q 0
    h✝ : Nontrivial R
    ⊢ Eq (AddMonoidAlgebra.supDegree D (HMul.hMul p q)) (HAdd.hAdd (AddMonoidAlgeb …
  -/
  apply supDegree_eq_of_max
    /-
      case inr.hb
      R : Type u_1
      A : Type u_3
      B : Type u_5
      inst✝⁶ : Semiring R
      inst✝⁵ : LinearOrder B
      inst✝⁴ : OrderBot B
      p q : AddMonoidAlgebra R A
      D : A → B
      inst✝³ : AddZeroClass A
      inst✝² : Add B
      inst✝¹ : AddLeftStrictMono B
      inst✝ : AddRightStrictMono B
      hD : Function.Injective D
      hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
      hpq : Ne (HMul.hMul (AddMonoidAlgebra.leadingCoeff D p) (AddMonoidAlgebra.lead …
      hp : Ne p 0
      hq : Ne q 0
      h✝ : Nontrivial R
      ⊢ Membership.mem (Set.range D) (HAdd.hAdd (AddMonoidAlgebra.supDegree D p) (Ad …
    -/
  · rw [← AddSubsemigroup.coe_set_mk (Set.range D), ← AddHom.srange_mk _ hadd, SetLike.mem_coe]
    /-
      case inr.hb
      R : Type u_1
      A : Type u_3
      B : Type u_5
      inst✝⁶ : Semiring R
      inst✝⁵ : LinearOrder B
      inst✝⁴ : OrderBot B
      p q : AddMonoidAlgebra R A
      D : A → B
      inst✝³ : AddZeroClass A
      inst✝² : Add B
      inst✝¹ : AddLeftStrictMono B
      inst✝ : AddRightStrictMono B
      hD : Function.Injective D
      hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
      hpq : Ne (HMul.hMul (AddMonoidAlgebra.leadingCoeff D p) (AddMonoidAlgebra.lead …
      hp : Ne p 0
      hq : Ne q 0
      h✝ : Nontrivial R
      ⊢ Membership.mem { toFun := D, map_add' := hadd }.srange (HAdd.hAdd (AddMonoid …
    -/
    exact add_mem (supDegree_mem_range D hp) (supDegree_mem_range D hq)
    /-
      🎉 no goals
    -/
    /-
      case inr.hmem
      R : Type u_1
      A : Type u_3
      B : Type u_5
      inst✝⁶ : Semiring R
      inst✝⁵ : LinearOrder B
      inst✝⁴ : OrderBot B
      p q : AddMonoidAlgebra R A
      D : A → B
      inst✝³ : AddZeroClass A
      inst✝² : Add B
      inst✝¹ : AddLeftStrictMono B
      inst✝ : AddRightStrictMono B
      hD : Function.Injective D
      hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
      hpq : Ne (HMul.hMul (AddMonoidAlgebra.leadingCoeff D p) (AddMonoidAlgebra.lead …
      hp : Ne p 0
      hq : Ne q 0
      h✝ : Nontrivial R
      ⊢ Membership.mem (HMul.hMul p q).support (Function.invFun D (HAdd.hAdd (AddMon …
    -/
  · simp_rw [Finsupp.mem_support_iff, apply_supDegree_add_supDegree hD hadd]
    /-
      case inr.hmem
      R : Type u_1
      A : Type u_3
      B : Type u_5
      inst✝⁶ : Semiring R
      inst✝⁵ : LinearOrder B
      inst✝⁴ : OrderBot B
      p q : AddMonoidAlgebra R A
      D : A → B
      inst✝³ : AddZeroClass A
      inst✝² : Add B
      inst✝¹ : AddLeftStrictMono B
      inst✝ : AddRightStrictMono B
      hD : Function.Injective D
      hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
      hpq : Ne (HMul.hMul (AddMonoidAlgebra.leadingCoeff D p) (AddMonoidAlgebra.lead …
      hp : Ne p 0
      hq : Ne q 0
      h✝ : Nontrivial R
      ⊢ Ne (HMul.hMul (AddMonoidAlgebra.leadingCoeff D p) (AddMonoidAlgebra.leadingC …
    -/
    exact hpq
    /-
      🎉 no goals
    -/
    /-
      case inr.hmax
      R : Type u_1
      A : Type u_3
      B : Type u_5
      inst✝⁶ : Semiring R
      inst✝⁵ : LinearOrder B
      inst✝⁴ : OrderBot B
      p q : AddMonoidAlgebra R A
      D : A → B
      inst✝³ : AddZeroClass A
      inst✝² : Add B
      inst✝¹ : AddLeftStrictMono B
      inst✝ : AddRightStrictMono B
      hD : Function.Injective D
      hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
      hpq : Ne (HMul.hMul (AddMonoidAlgebra.leadingCoeff D p) (AddMonoidAlgebra.lead …
      hp : Ne p 0
      hq : Ne q 0
      h✝ : Nontrivial R
      ⊢ ∀ (a : A), Membership.mem (HMul.hMul p q).support a → LE.le (D a) (HAdd.hAdd …
    -/
  · have := covariantClass_le_of_lt B B (· + ·)
    /-
      case inr.hmax
      R : Type u_1
      A : Type u_3
      B : Type u_5
      inst✝⁶ : Semiring R
      inst✝⁵ : LinearOrder B
      inst✝⁴ : OrderBot B
      p q : AddMonoidAlgebra R A
      D : A → B
      inst✝³ : AddZeroClass A
      inst✝² : Add B
      inst✝¹ : AddLeftStrictMono B
      inst✝ : AddRightStrictMono B
      hD : Function.Injective D
      hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
      hpq : Ne (HMul.hMul (AddMonoidAlgebra.leadingCoeff D p) (AddMonoidAlgebra.lead …
      hp : Ne p 0
      hq : Ne q 0
      h✝ : Nontrivial R
      this : CovariantClass B B (fun x1 x2 => HAdd.hAdd x1 x2) fun x1 x2 => LE.le x1 …
      ⊢ ∀ (a : A), Membership.mem (HMul.hMul p q).support a → LE.le (D a) (HAdd.hAdd …
    -/
    have := covariantClass_le_of_lt B B (Function.swap (· + ·))
    /-
      case inr.hmax
      R : Type u_1
      A : Type u_3
      B : Type u_5
      inst✝⁶ : Semiring R
      inst✝⁵ : LinearOrder B
      inst✝⁴ : OrderBot B
      p q : AddMonoidAlgebra R A
      D : A → B
      inst✝³ : AddZeroClass A
      inst✝² : Add B
      inst✝¹ : AddLeftStrictMono B
      inst✝ : AddRightStrictMono B
      hD : Function.Injective D
      hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
      hpq : Ne (HMul.hMul (AddMonoidAlgebra.leadingCoeff D p) (AddMonoidAlgebra.lead …
      hp : Ne p 0
      hq : Ne q 0
      h✝ : Nontrivial R
      this✝ : CovariantClass B B (fun x1 x2 => HAdd.hAdd x1 x2) fun x1 x2 => LE.le x …
      this : CovariantClass B B (Function.swap fun x1 x2 => HAdd.hAdd x1 x2) fun x1  …
      ⊢ ∀ (a : A), Membership.mem (HMul.hMul p q).support a → LE.le (D a) (HAdd.hAdd …
    -/
    exact fun a ha => (Finset.le_sup ha).trans (supDegree_mul_le hadd)
    /-
      🎉 no goals
    -/


lemma Monic.supDegree_mul_of_ne_zero_left
    (hD : D.Injective) (hadd : ∀ a1 a2, D (a1 + a2) = D a1 + D a2)
    (hq : q.Monic D) (hp : p ≠ 0) :
    (p * q).supDegree D = p.supDegree D + q.supDegree D := by
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : LinearOrder B
    inst✝⁴ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝³ : AddZeroClass A
    inst✝² : Add B
    inst✝¹ : AddLeftStrictMono B
    inst✝ : AddRightStrictMono B
    hD : Function.Injective D
    hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
    hq : AddMonoidAlgebra.Monic D q
    hp : Ne p 0
    ⊢ Eq (AddMonoidAlgebra.supDegree D (HMul.hMul p q)) (HAdd.hAdd (AddMonoidAlgeb …
  -/
  cases subsingleton_or_nontrivial R; · exact (hp (Subsingleton.elim _ _)).elim
                                        /-
                                          🎉 no goals
                                        -/
  /-
    case inr
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : LinearOrder B
    inst✝⁴ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝³ : AddZeroClass A
    inst✝² : Add B
    inst✝¹ : AddLeftStrictMono B
    inst✝ : AddRightStrictMono B
    hD : Function.Injective D
    hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
    hq : AddMonoidAlgebra.Monic D q
    hp : Ne p 0
    h✝ : Nontrivial R
    ⊢ Eq (AddMonoidAlgebra.supDegree D (HMul.hMul p q)) (HAdd.hAdd (AddMonoidAlgeb …
  -/
  apply supDegree_mul hD hadd ?_ hp hq.ne_zero
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : LinearOrder B
    inst✝⁴ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝³ : AddZeroClass A
    inst✝² : Add B
    inst✝¹ : AddLeftStrictMono B
    inst✝ : AddRightStrictMono B
    hD : Function.Injective D
    hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
    hq : AddMonoidAlgebra.Monic D q
    hp : Ne p 0
    h✝ : Nontrivial R
    ⊢ Ne (HMul.hMul (AddMonoidAlgebra.leadingCoeff D p) (AddMonoidAlgebra.leadingC …
  -/
  simp_rw [hq, mul_one, Ne, leadingCoeff_eq_zero hD, hp, not_false_eq_true]
  /-
    🎉 no goals
  -/


lemma Monic.supDegree_mul_of_ne_zero_right
    (hD : D.Injective) (hadd : ∀ a1 a2, D (a1 + a2) = D a1 + D a2)
    (hp : p.Monic D) (hq : q ≠ 0) :
    (p * q).supDegree D = p.supDegree D + q.supDegree D := by
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : LinearOrder B
    inst✝⁴ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝³ : AddZeroClass A
    inst✝² : Add B
    inst✝¹ : AddLeftStrictMono B
    inst✝ : AddRightStrictMono B
    hD : Function.Injective D
    hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
    hp : AddMonoidAlgebra.Monic D p
    hq : Ne q 0
    ⊢ Eq (AddMonoidAlgebra.supDegree D (HMul.hMul p q)) (HAdd.hAdd (AddMonoidAlgeb …
  -/
  cases subsingleton_or_nontrivial R; · exact (hq (Subsingleton.elim _ _)).elim
                                        /-
                                          🎉 no goals
                                        -/
  /-
    case inr
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : LinearOrder B
    inst✝⁴ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝³ : AddZeroClass A
    inst✝² : Add B
    inst✝¹ : AddLeftStrictMono B
    inst✝ : AddRightStrictMono B
    hD : Function.Injective D
    hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
    hp : AddMonoidAlgebra.Monic D p
    hq : Ne q 0
    h✝ : Nontrivial R
    ⊢ Eq (AddMonoidAlgebra.supDegree D (HMul.hMul p q)) (HAdd.hAdd (AddMonoidAlgeb …
  -/
  apply supDegree_mul hD hadd ?_ hp.ne_zero hq
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : LinearOrder B
    inst✝⁴ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝³ : AddZeroClass A
    inst✝² : Add B
    inst✝¹ : AddLeftStrictMono B
    inst✝ : AddRightStrictMono B
    hD : Function.Injective D
    hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
    hp : AddMonoidAlgebra.Monic D p
    hq : Ne q 0
    h✝ : Nontrivial R
    ⊢ Ne (HMul.hMul (AddMonoidAlgebra.leadingCoeff D p) (AddMonoidAlgebra.leadingC …
  -/
  simp_rw [hp, one_mul, Ne, leadingCoeff_eq_zero hD, hq, not_false_eq_true]
  /-
    🎉 no goals
  -/


lemma Monic.supDegree_mul
    (hD : D.Injective) (hadd : ∀ a1 a2, D (a1 + a2) = D a1 + D a2)
    (hbot : (⊥ : B) + ⊥ = ⊥) (hp : p.Monic D) (hq : q.Monic D) :
    (p * q).supDegree D = p.supDegree D + q.supDegree D := by
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : LinearOrder B
    inst✝⁴ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝³ : AddZeroClass A
    inst✝² : Add B
    inst✝¹ : AddLeftStrictMono B
    inst✝ : AddRightStrictMono B
    hD : Function.Injective D
    hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
    hbot : Eq (HAdd.hAdd Bot.bot Bot.bot) Bot.bot
    hp : AddMonoidAlgebra.Monic D p
    hq : AddMonoidAlgebra.Monic D q
    ⊢ Eq (AddMonoidAlgebra.supDegree D (HMul.hMul p q)) (HAdd.hAdd (AddMonoidAlgeb …
  -/
  cases subsingleton_or_nontrivial R
    /-
      case inl
      R : Type u_1
      A : Type u_3
      B : Type u_5
      inst✝⁶ : Semiring R
      inst✝⁵ : LinearOrder B
      inst✝⁴ : OrderBot B
      p q : AddMonoidAlgebra R A
      D : A → B
      inst✝³ : AddZeroClass A
      inst✝² : Add B
      inst✝¹ : AddLeftStrictMono B
      inst✝ : AddRightStrictMono B
      hD : Function.Injective D
      hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
      hbot : Eq (HAdd.hAdd Bot.bot Bot.bot) Bot.bot
      hp : AddMonoidAlgebra.Monic D p
      hq : AddMonoidAlgebra.Monic D q
      h✝ : Subsingleton R
      ⊢ Eq (AddMonoidAlgebra.supDegree D (HMul.hMul p q)) (HAdd.hAdd (AddMonoidAlgeb …
    -/
  · simp_rw [Subsingleton.eq_zero p, Subsingleton.eq_zero q, mul_zero, supDegree_zero, hbot]
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : LinearOrder B
    inst✝⁴ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝³ : AddZeroClass A
    inst✝² : Add B
    inst✝¹ : AddLeftStrictMono B
    inst✝ : AddRightStrictMono B
    hD : Function.Injective D
    hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
    hbot : Eq (HAdd.hAdd Bot.bot Bot.bot) Bot.bot
    hp : AddMonoidAlgebra.Monic D p
    hq : AddMonoidAlgebra.Monic D q
    h✝ : Nontrivial R
    ⊢ Eq (AddMonoidAlgebra.supDegree D (HMul.hMul p q)) (HAdd.hAdd (AddMonoidAlgeb …
  -/
  exact hq.supDegree_mul_of_ne_zero_left hD hadd hp.ne_zero
  /-
    🎉 no goals
  -/


lemma leadingCoeff_mul [NoZeroDivisors R]
    (hD : D.Injective) (hadd : ∀ a1 a2, D (a1 + a2) = D a1 + D a2) :
    (p * q).leadingCoeff D = p.leadingCoeff D * q.leadingCoeff D := by
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁷ : Semiring R
    inst✝⁶ : LinearOrder B
    inst✝⁵ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝⁴ : AddZeroClass A
    inst✝³ : Add B
    inst✝² : AddLeftStrictMono B
    inst✝¹ : AddRightStrictMono B
    inst✝ : NoZeroDivisors R
    hD : Function.Injective D
    hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
    ⊢ Eq (AddMonoidAlgebra.leadingCoeff D (HMul.hMul p q)) (HMul.hMul (AddMonoidAl …
  -/
  obtain rfl | hp := eq_or_ne p 0
    /-
      case inl
      R : Type u_1
      A : Type u_3
      B : Type u_5
      inst✝⁷ : Semiring R
      inst✝⁶ : LinearOrder B
      inst✝⁵ : OrderBot B
      q : AddMonoidAlgebra R A
      D : A → B
      inst✝⁴ : AddZeroClass A
      inst✝³ : Add B
      inst✝² : AddLeftStrictMono B
      inst✝¹ : AddRightStrictMono B
      inst✝ : NoZeroDivisors R
      hD : Function.Injective D
      hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
      ⊢ Eq (AddMonoidAlgebra.leadingCoeff D (HMul.hMul 0 q)) (HMul.hMul (AddMonoidAl …
    -/
  · simp_rw [leadingCoeff_zero, zero_mul, leadingCoeff_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁷ : Semiring R
    inst✝⁶ : LinearOrder B
    inst✝⁵ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝⁴ : AddZeroClass A
    inst✝³ : Add B
    inst✝² : AddLeftStrictMono B
    inst✝¹ : AddRightStrictMono B
    inst✝ : NoZeroDivisors R
    hD : Function.Injective D
    hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
    hp : Ne p 0
    ⊢ Eq (AddMonoidAlgebra.leadingCoeff D (HMul.hMul p q)) (HMul.hMul (AddMonoidAl …
  -/
  obtain rfl | hq := eq_or_ne q 0
    /-
      case inr.inl
      R : Type u_1
      A : Type u_3
      B : Type u_5
      inst✝⁷ : Semiring R
      inst✝⁶ : LinearOrder B
      inst✝⁵ : OrderBot B
      p : AddMonoidAlgebra R A
      D : A → B
      inst✝⁴ : AddZeroClass A
      inst✝³ : Add B
      inst✝² : AddLeftStrictMono B
      inst✝¹ : AddRightStrictMono B
      inst✝ : NoZeroDivisors R
      hD : Function.Injective D
      hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
      hp : Ne p 0
      ⊢ Eq (AddMonoidAlgebra.leadingCoeff D (HMul.hMul p 0)) (HMul.hMul (AddMonoidAl …
    -/
  · simp_rw [leadingCoeff_zero, mul_zero, leadingCoeff_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁷ : Semiring R
    inst✝⁶ : LinearOrder B
    inst✝⁵ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝⁴ : AddZeroClass A
    inst✝³ : Add B
    inst✝² : AddLeftStrictMono B
    inst✝¹ : AddRightStrictMono B
    inst✝ : NoZeroDivisors R
    hD : Function.Injective D
    hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
    hp : Ne p 0
    hq : Ne q 0
    ⊢ Eq (AddMonoidAlgebra.leadingCoeff D (HMul.hMul p q)) (HMul.hMul (AddMonoidAl …
  -/
  rw [← apply_supDegree_add_supDegree hD hadd, ← supDegree_mul hD hadd ?_ hp hq, leadingCoeff]
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁷ : Semiring R
    inst✝⁶ : LinearOrder B
    inst✝⁵ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝⁴ : AddZeroClass A
    inst✝³ : Add B
    inst✝² : AddLeftStrictMono B
    inst✝¹ : AddRightStrictMono B
    inst✝ : NoZeroDivisors R
    hD : Function.Injective D
    hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
    hp : Ne p 0
    hq : Ne q 0
    ⊢ Ne (HMul.hMul (AddMonoidAlgebra.leadingCoeff D p) (AddMonoidAlgebra.leadingC …
  -/
                        /-
                          🎉 no goals
                        -/
  apply mul_ne_zero <;> rwa [Ne, leadingCoeff_eq_zero hD]
                        /-
                          🎉 no goals
                        -/


lemma Monic.leadingCoeff_mul_eq_left
    (hD : D.Injective) (hadd : ∀ a1 a2, D (a1 + a2) = D a1 + D a2) (hq : q.Monic D) :
    (p * q).leadingCoeff D = p.leadingCoeff D := by
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : LinearOrder B
    inst✝⁴ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝³ : AddZeroClass A
    inst✝² : Add B
    inst✝¹ : AddLeftStrictMono B
    inst✝ : AddRightStrictMono B
    hD : Function.Injective D
    hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
    hq : AddMonoidAlgebra.Monic D q
    ⊢ Eq (AddMonoidAlgebra.leadingCoeff D (HMul.hMul p q)) (AddMonoidAlgebra.leadi …
  -/
  obtain rfl | hp := eq_or_ne p 0
    /-
      case inl
      R : Type u_1
      A : Type u_3
      B : Type u_5
      inst✝⁶ : Semiring R
      inst✝⁵ : LinearOrder B
      inst✝⁴ : OrderBot B
      q : AddMonoidAlgebra R A
      D : A → B
      inst✝³ : AddZeroClass A
      inst✝² : Add B
      inst✝¹ : AddLeftStrictMono B
      inst✝ : AddRightStrictMono B
      hD : Function.Injective D
      hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
      hq : AddMonoidAlgebra.Monic D q
      ⊢ Eq (AddMonoidAlgebra.leadingCoeff D (HMul.hMul 0 q)) (AddMonoidAlgebra.leadi …
    -/
  · rw [zero_mul]
    /-
      🎉 no goals
    -/
  rw [leadingCoeff, hq.supDegree_mul_of_ne_zero_left hD hadd hp,
    apply_supDegree_add_supDegree hD hadd, hq, mul_one]


lemma Monic.leadingCoeff_mul_eq_right
    (hD : D.Injective) (hadd : ∀ a1 a2, D (a1 + a2) = D a1 + D a2) (hp : p.Monic D) :
    (p * q).leadingCoeff D = q.leadingCoeff D := by
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : LinearOrder B
    inst✝⁴ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝³ : AddZeroClass A
    inst✝² : Add B
    inst✝¹ : AddLeftStrictMono B
    inst✝ : AddRightStrictMono B
    hD : Function.Injective D
    hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
    hp : AddMonoidAlgebra.Monic D p
    ⊢ Eq (AddMonoidAlgebra.leadingCoeff D (HMul.hMul p q)) (AddMonoidAlgebra.leadi …
  -/
  obtain rfl | hq := eq_or_ne q 0
    /-
      case inl
      R : Type u_1
      A : Type u_3
      B : Type u_5
      inst✝⁶ : Semiring R
      inst✝⁵ : LinearOrder B
      inst✝⁴ : OrderBot B
      p : AddMonoidAlgebra R A
      D : A → B
      inst✝³ : AddZeroClass A
      inst✝² : Add B
      inst✝¹ : AddLeftStrictMono B
      inst✝ : AddRightStrictMono B
      hD : Function.Injective D
      hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
      hp : AddMonoidAlgebra.Monic D p
      ⊢ Eq (AddMonoidAlgebra.leadingCoeff D (HMul.hMul p 0)) (AddMonoidAlgebra.leadi …
    -/
  · rw [mul_zero]
    /-
      🎉 no goals
    -/
  rw [leadingCoeff, hp.supDegree_mul_of_ne_zero_right hD hadd hq,
    apply_supDegree_add_supDegree hD hadd, hp, one_mul]


lemma Monic.mul
    (hD : D.Injective) (hadd : ∀ a1 a2, D (a1 + a2) = D a1 + D a2)
    (hp : p.Monic D) (hq : q.Monic D) : (p * q).Monic D := by
  /-
    R : Type u_1
    A : Type u_3
    B : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : LinearOrder B
    inst✝⁴ : OrderBot B
    p q : AddMonoidAlgebra R A
    D : A → B
    inst✝³ : AddZeroClass A
    inst✝² : Add B
    inst✝¹ : AddLeftStrictMono B
    inst✝ : AddRightStrictMono B
    hD : Function.Injective D
    hadd : ∀ (a1 a2 : A), Eq (D (HAdd.hAdd a1 a2)) (HAdd.hAdd (D a1) (D a2))
    hp : AddMonoidAlgebra.Monic D p
    hq : AddMonoidAlgebra.Monic D q
    ⊢ AddMonoidAlgebra.Monic D (HMul.hMul p q)
  -/
  rw [Monic, hq.leadingCoeff_mul_eq_left hD hadd]; exact hp
                                                   /-
                                                     🎉 no goals
                                                   -/


lemma Monic.pow
    (hadd : ∀ a1 a2, D (a1 + a2) = D a1 + D a2) (hD : D.Injective)
    (hp : p.Monic D) : (p ^ n).Monic D := by
  induction n with
  | zero => rw [pow_zero]; exact monic_one hD
  | succ n ih => rw [pow_succ']; exact hp.mul hD hadd ih


lemma Monic.supDegree_pow
    (hzero : D 0 = 0) (hadd : ∀ a1 a2, D (a1 + a2) = D a1 + D a2) (hD : D.Injective)
    [Nontrivial R] (hp : p.Monic D) :
    (p ^ n).supDegree D = n • p.supDegree D := by
  induction n with
  | zero => rw [pow_zero, zero_nsmul, one_def, supDegree_single 0 1, if_neg one_ne_zero, hzero]
  | succ n ih => rw [pow_succ', (hp.pow hadd hD).supDegree_mul_of_ne_zero_left hD hadd hp.ne_zero,
      ih, succ_nsmul']


/-- Let `R` be a semiring, let `A` be an `AddZeroClass`, let `T` be an `OrderTop`,
and let `D : A → T` be a "degree" function.
For an element `f : R[A]`, the element `infDegree f : T` is the infimum of all the elements in the
support of `f`, or `⊤` if `f` is zero.
Often, the Type `T` is `WithTop A`,
If, further, `A` has a linear order, then this notion coincides with the usual one,
using the minimum of the exponents. -/
abbrev infDegree (f : R[A]) : T :=
  f.support.inf D


theorem le_infDegree_add (f g : R[A]) :
    (f.infDegree D) ⊓ (g.infDegree D) ≤ (f + g).infDegree D :=
  le_inf_support_add D f g


variable {D} in
theorem infDegree_withTop_some_comp {s : AddMonoidAlgebra R A} (hs : s.support.Nonempty) :
    infDegree (WithTop.some ∘ D) s = infDegree D s := by
  /-
    R : Type u_1
    A : Type u_3
    T : Type u_4
    inst✝² : Semiring R
    inst✝¹ : SemilatticeInf T
    inst✝ : OrderTop T
    D : A → T
    s : AddMonoidAlgebra R A
    hs : s.support.Nonempty
    ⊢ Eq (AddMonoidAlgebra.infDegree (Function.comp WithTop.some D) s) ↑(AddMonoid …
  -/
  unfold AddMonoidAlgebra.infDegree
  /-
    R : Type u_1
    A : Type u_3
    T : Type u_4
    inst✝² : Semiring R
    inst✝¹ : SemilatticeInf T
    inst✝ : OrderTop T
    D : A → T
    s : AddMonoidAlgebra R A
    hs : s.support.Nonempty
    ⊢ Eq (s.support.inf (Function.comp WithTop.some D)) ↑(s.support.inf D)
  -/
  rw [← Finset.coe_inf' hs, Finset.inf'_eq_inf]
  /-
    🎉 no goals
  -/


theorem le_infDegree_mul [AddZeroClass A] [Add T] [AddLeftMono T] [AddRightMono T]
    (D : AddHom A T) (f g : R[A]) :
    f.infDegree D + g.infDegree D ≤ (f * g).infDegree D :=
  le_inf_support_mul (fun {a b : A} => (map_add D a b).ge) _ _


