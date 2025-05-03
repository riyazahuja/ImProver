/-- `Finset.noncommProd` is “injective” in `f` if `f` maps into independent subgroups.  This
generalizes (one direction of) `Subgroup.disjoint_iff_mul_eq_one`. -/
@[to_additive "`Finset.noncommSum` is “injective” in `f` if `f` maps into independent subgroups.
This generalizes (one direction of) `AddSubgroup.disjoint_iff_add_eq_zero`. "]
theorem eq_one_of_noncommProd_eq_one_of_iSupIndep {ι : Type*} (s : Finset ι) (f : ι → G) (comm)
    (K : ι → Subgroup G) (hind : iSupIndep K) (hmem : ∀ x ∈ s, f x ∈ K x)
    (heq1 : s.noncommProd f comm = 1) : ∀ i ∈ s, f i = 1 := by
  classical
    revert heq1
    induction' s using Finset.induction_on with i s hnmem ih
    · simp
    · have hcomm := comm.mono (Finset.coe_subset.2 <| Finset.subset_insert _ _)
      simp only [Finset.forall_mem_insert] at hmem
      have hmem_bsupr : s.noncommProd f hcomm ∈ ⨆ i ∈ (s : Set ι), K i := by
        refine Subgroup.noncommProd_mem _ _ ?_
        intro x hx
        have : K x ≤ ⨆ i ∈ (s : Set ι), K i := le_iSup₂ (f := fun i _ => K i) x hx
        exact this (hmem.2 x hx)
      intro heq1
      rw [Finset.noncommProd_insert_of_not_mem _ _ _ _ hnmem] at heq1
      have hnmem' : i ∉ (s : Set ι) := by simpa
      obtain ⟨heq1i : f i = 1, heq1S : s.noncommProd f _ = 1⟩ :=
        Subgroup.disjoint_iff_mul_eq_one.mp (hind.disjoint_biSup hnmem') hmem.1 hmem_bsupr heq1
      intro i h
      simp only [Finset.mem_insert] at h
      rcases h with (rfl | h)
      · exact heq1i
      · refine ih hcomm hmem.2 heq1S _ h


@[deprecated (since := "2024-11-24")]
alias eq_one_of_noncommProd_eq_one_of_independent := eq_one_of_noncommProd_eq_one_of_iSupIndep


/-- The canonical homomorphism from a family of monoids. -/
@[to_additive "The canonical homomorphism from a family of additive monoids. See also
`LinearMap.lsum` for a linear version without the commutativity assumption."]
def noncommPiCoprod : (∀ i : ι, N i) →* M where
  toFun f := Finset.univ.noncommProd (fun i => ϕ i (f i)) fun _ _ _ _ h => hcomm h _ _
  map_one' := by
    /-
      M : Type u_1
      inst✝² : Monoid M
      ι : Type u_2
      inst✝¹ : Fintype ι
      N : ι → Type u_3
      inst✝ : (i : ι) → Monoid (N i)
      ϕ : (i : ι) → MonoidHom (N i) M
      hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
      ⊢ Eq ((fun f => Finset.univ.noncommProd (fun i => (ϕ i) (f i)) ⋯) 1) 1
    -/
    apply (Finset.noncommProd_eq_pow_card _ _ _ _ _).trans (one_pow _)
    /-
      M : Type u_1
      inst✝² : Monoid M
      ι : Type u_2
      inst✝¹ : Fintype ι
      N : ι → Type u_3
      inst✝ : (i : ι) → Monoid (N i)
      ϕ : (i : ι) → MonoidHom (N i) M
      hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
      ⊢ ∀ (x : ι), Membership.mem Finset.univ x → Eq ((ϕ x) (1 x)) 1
    -/
    simp
    /-
      🎉 no goals
    -/
  map_mul' f g := by
    classical
    simp only
    convert @Finset.noncommProd_mul_distrib _ _ _ _ (fun i => ϕ i (f i)) (fun i => ϕ i (g i)) _ _ _
    · exact map_mul _ _ _
    · rintro i - j - h
      exact hcomm h _ _


@[to_additive (attr := simp)]
theorem noncommPiCoprod_mulSingle [DecidableEq ι] (i : ι) (y : N i) :
    noncommPiCoprod ϕ hcomm (Pi.mulSingle i y) = ϕ i y := by
  change Finset.univ.noncommProd (fun j => ϕ j (Pi.mulSingle i y j)) (fun _ _ _ _ h => hcomm h _ _)
    = ϕ i y
  /-
    M : Type u_1
    inst✝³ : Monoid M
    ι : Type u_2
    inst✝² : Fintype ι
    N : ι → Type u_3
    inst✝¹ : (i : ι) → Monoid (N i)
    ϕ : (i : ι) → MonoidHom (N i) M
    hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
    inst✝ : DecidableEq ι
    i : ι
    y : N i
    ⊢ Eq (Finset.univ.noncommProd (fun j => (ϕ j) (Pi.mulSingle i y j)) ⋯) ((ϕ i) y)
  -/
  rw [← Finset.insert_erase (Finset.mem_univ i)]
  /-
    M : Type u_1
    inst✝³ : Monoid M
    ι : Type u_2
    inst✝² : Fintype ι
    N : ι → Type u_3
    inst✝¹ : (i : ι) → Monoid (N i)
    ϕ : (i : ι) → MonoidHom (N i) M
    hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
    inst✝ : DecidableEq ι
    i : ι
    y : N i
    ⊢ Eq ((Insert.insert i (Finset.univ.erase i)).noncommProd (fun j => (ϕ j) (Pi. …
  -/
  rw [Finset.noncommProd_insert_of_not_mem _ _ _ _ (Finset.not_mem_erase i _)]
  /-
    M : Type u_1
    inst✝³ : Monoid M
    ι : Type u_2
    inst✝² : Fintype ι
    N : ι → Type u_3
    inst✝¹ : (i : ι) → Monoid (N i)
    ϕ : (i : ι) → MonoidHom (N i) M
    hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
    inst✝ : DecidableEq ι
    i : ι
    y : N i
    ⊢ Eq (HMul.hMul ((ϕ i) (Pi.mulSingle i y i)) ((Finset.univ.erase i).noncommPro …
  -/
  rw [Pi.mulSingle_eq_same]
  /-
    M : Type u_1
    inst✝³ : Monoid M
    ι : Type u_2
    inst✝² : Fintype ι
    N : ι → Type u_3
    inst✝¹ : (i : ι) → Monoid (N i)
    ϕ : (i : ι) → MonoidHom (N i) M
    hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
    inst✝ : DecidableEq ι
    i : ι
    y : N i
    ⊢ Eq (HMul.hMul ((ϕ i) y) ((Finset.univ.erase i).noncommProd (fun j => (ϕ j) ( …
  -/
  rw [Finset.noncommProd_eq_pow_card]
    /-
      M : Type u_1
      inst✝³ : Monoid M
      ι : Type u_2
      inst✝² : Fintype ι
      N : ι → Type u_3
      inst✝¹ : (i : ι) → Monoid (N i)
      ϕ : (i : ι) → MonoidHom (N i) M
      hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
      inst✝ : DecidableEq ι
      i : ι
      y : N i
      ⊢ Eq (HMul.hMul ((ϕ i) y) (HPow.hPow ?m (Finset.univ.erase i).card)) ((ϕ i) y)
    -/
  · rw [one_pow]
    /-
      M : Type u_1
      inst✝³ : Monoid M
      ι : Type u_2
      inst✝² : Fintype ι
      N : ι → Type u_3
      inst✝¹ : (i : ι) → Monoid (N i)
      ϕ : (i : ι) → MonoidHom (N i) M
      hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
      inst✝ : DecidableEq ι
      i : ι
      y : N i
      ⊢ Eq (HMul.hMul ((ϕ i) y) 1) ((ϕ i) y)
    -/
    exact mul_one _
    /-
      🎉 no goals
    -/
    /-
      case h
      M : Type u_1
      inst✝³ : Monoid M
      ι : Type u_2
      inst✝² : Fintype ι
      N : ι → Type u_3
      inst✝¹ : (i : ι) → Monoid (N i)
      ϕ : (i : ι) → MonoidHom (N i) M
      hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
      inst✝ : DecidableEq ι
      i : ι
      y : N i
      ⊢ ∀ (x : ι), Membership.mem (Finset.univ.erase i) x → Eq ((ϕ x) (Pi.mulSingle  …
    -/
  · intro j hj
    /-
      case h
      M : Type u_1
      inst✝³ : Monoid M
      ι : Type u_2
      inst✝² : Fintype ι
      N : ι → Type u_3
      inst✝¹ : (i : ι) → Monoid (N i)
      ϕ : (i : ι) → MonoidHom (N i) M
      hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
      inst✝ : DecidableEq ι
      i : ι
      y : N i
      j : ι
      hj : Membership.mem (Finset.univ.erase i) j
      ⊢ Eq ((ϕ j) (Pi.mulSingle i y j)) 1
    -/
    simp only [Finset.mem_erase] at hj
    /-
      case h
      M : Type u_1
      inst✝³ : Monoid M
      ι : Type u_2
      inst✝² : Fintype ι
      N : ι → Type u_3
      inst✝¹ : (i : ι) → Monoid (N i)
      ϕ : (i : ι) → MonoidHom (N i) M
      hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
      inst✝ : DecidableEq ι
      i : ι
      y : N i
      j : ι
      hj : And (Ne j i) (Membership.mem Finset.univ j)
      ⊢ Eq ((ϕ j) (Pi.mulSingle i y j)) 1
    -/
    simp [hj]
    /-
      🎉 no goals
    -/


/-- The universal property of `MonoidHom.noncommPiCoprod` -/
@[to_additive "The universal property of `AddMonoidHom.noncommPiCoprod`"]
def noncommPiCoprodEquiv [DecidableEq ι] :
    { ϕ : ∀ i, N i →* M // Pairwise fun i j => ∀ x y, Commute (ϕ i x) (ϕ j y) } ≃
      ((∀ i, N i) →* M) where
  toFun ϕ := noncommPiCoprod ϕ.1 ϕ.2
  invFun f :=
    ⟨fun i => f.comp (MonoidHom.mulSingle N i), fun _ _ hij x y =>
      Commute.map (Pi.mulSingle_commute hij x y) f⟩
  left_inv ϕ := by
    /-
      M : Type u_1
      inst✝³ : Monoid M
      ι : Type u_2
      inst✝² : Fintype ι
      N : ι → Type u_3
      inst✝¹ : (i : ι) → Monoid (N i)
      ϕ✝ : (i : ι) → MonoidHom (N i) M
      hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ✝ i) x) ((ϕ✝ j) y)
      inst✝ : DecidableEq ι
      ϕ : Subtype fun ϕ => Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) …
      ⊢ Eq ((fun f => ⟨fun i => f.comp (MonoidHom.mulSingle N i), ⋯⟩) ((fun ϕ => Mon …
    -/
    ext
    /-
      case a.h.h
      M : Type u_1
      inst✝³ : Monoid M
      ι : Type u_2
      inst✝² : Fintype ι
      N : ι → Type u_3
      inst✝¹ : (i : ι) → Monoid (N i)
      ϕ✝ : (i : ι) → MonoidHom (N i) M
      hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ✝ i) x) ((ϕ✝ j) y)
      inst✝ : DecidableEq ι
      ϕ : Subtype fun ϕ => Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) …
      x✝¹ : ι
      x✝ : N x✝¹
      ⊢ Eq ((↑((fun f => ⟨fun i => f.comp (MonoidHom.mulSingle N i), ⋯⟩) ((fun ϕ =>  …
    -/
    simp only [coe_comp, Function.comp_apply, mulSingle_apply, noncommPiCoprod_mulSingle]
    /-
      🎉 no goals
    -/
  right_inv f := pi_ext fun i x => by
    /-
      M : Type u_1
      inst✝³ : Monoid M
      ι : Type u_2
      inst✝² : Fintype ι
      N : ι → Type u_3
      inst✝¹ : (i : ι) → Monoid (N i)
      ϕ : (i : ι) → MonoidHom (N i) M
      hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
      inst✝ : DecidableEq ι
      f : MonoidHom ((i : ι) → N i) M
      i : ι
      x : N i
      ⊢ Eq (((fun ϕ => MonoidHom.noncommPiCoprod ↑ϕ ⋯) ((fun f => ⟨fun i => f.comp ( …
    -/
    simp only [noncommPiCoprod_mulSingle, coe_comp, Function.comp_apply, mulSingle_apply]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem noncommPiCoprod_mrange :
    MonoidHom.mrange (noncommPiCoprod ϕ hcomm) = ⨆ i : ι, MonoidHom.mrange (ϕ i) := by
  /-
    M : Type u_1
    inst✝² : Monoid M
    ι : Type u_2
    inst✝¹ : Fintype ι
    N : ι → Type u_3
    inst✝ : (i : ι) → Monoid (N i)
    ϕ : (i : ι) → MonoidHom (N i) M
    hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
    ⊢ Eq (MonoidHom.mrange (MonoidHom.noncommPiCoprod ϕ hcomm)) (iSup fun i => Mon …
  -/
  letI := Classical.decEq ι
  /-
    M : Type u_1
    inst✝² : Monoid M
    ι : Type u_2
    inst✝¹ : Fintype ι
    N : ι → Type u_3
    inst✝ : (i : ι) → Monoid (N i)
    ϕ : (i : ι) → MonoidHom (N i) M
    hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
    this : DecidableEq ι := Classical.decEq ι
    ⊢ Eq (MonoidHom.mrange (MonoidHom.noncommPiCoprod ϕ hcomm)) (iSup fun i => Mon …
  -/
  apply le_antisymm
    /-
      case a
      M : Type u_1
      inst✝² : Monoid M
      ι : Type u_2
      inst✝¹ : Fintype ι
      N : ι → Type u_3
      inst✝ : (i : ι) → Monoid (N i)
      ϕ : (i : ι) → MonoidHom (N i) M
      hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
      this : DecidableEq ι := Classical.decEq ι
      ⊢ LE.le (MonoidHom.mrange (MonoidHom.noncommPiCoprod ϕ hcomm)) (iSup fun i =>  …
    -/
  · rintro x ⟨f, rfl⟩
    /-
      case a.intro
      M : Type u_1
      inst✝² : Monoid M
      ι : Type u_2
      inst✝¹ : Fintype ι
      N : ι → Type u_3
      inst✝ : (i : ι) → Monoid (N i)
      ϕ : (i : ι) → MonoidHom (N i) M
      hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
      this : DecidableEq ι := Classical.decEq ι
      f : (i : ι) → N i
      ⊢ Membership.mem (iSup fun i => MonoidHom.mrange (ϕ i)) ((MonoidHom.noncommPiC …
    -/
    refine Submonoid.noncommProd_mem _ _ _ (fun _ _ _ _ h => hcomm h _ _) (fun i _ => ?_)
    /-
      case a.intro
      M : Type u_1
      inst✝² : Monoid M
      ι : Type u_2
      inst✝¹ : Fintype ι
      N : ι → Type u_3
      inst✝ : (i : ι) → Monoid (N i)
      ϕ : (i : ι) → MonoidHom (N i) M
      hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
      this : DecidableEq ι := Classical.decEq ι
      f : (i : ι) → N i
      i : ι
      x✝ : Membership.mem Finset.univ i
      ⊢ Membership.mem (iSup fun i => MonoidHom.mrange (ϕ i)) ((ϕ i) (f i))
    -/
    apply Submonoid.mem_sSup_of_mem
      /-
        case a.intro.hs
        M : Type u_1
        inst✝² : Monoid M
        ι : Type u_2
        inst✝¹ : Fintype ι
        N : ι → Type u_3
        inst✝ : (i : ι) → Monoid (N i)
        ϕ : (i : ι) → MonoidHom (N i) M
        hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
        this : DecidableEq ι := Classical.decEq ι
        f : (i : ι) → N i
        i : ι
        x✝ : Membership.mem Finset.univ i
        ⊢ Membership.mem (Set.range fun i => MonoidHom.mrange (ϕ i)) ?a.intro.s✝
      -/
    · use i
      /-
        🎉 no goals
      -/
    /-
      case a.intro.a
      M : Type u_1
      inst✝² : Monoid M
      ι : Type u_2
      inst✝¹ : Fintype ι
      N : ι → Type u_3
      inst✝ : (i : ι) → Monoid (N i)
      ϕ : (i : ι) → MonoidHom (N i) M
      hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
      this : DecidableEq ι := Classical.decEq ι
      f : (i : ι) → N i
      i : ι
      x✝ : Membership.mem Finset.univ i
      ⊢ Membership.mem ((fun i => MonoidHom.mrange (ϕ i)) i) ((ϕ i) (f i))
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case a
      M : Type u_1
      inst✝² : Monoid M
      ι : Type u_2
      inst✝¹ : Fintype ι
      N : ι → Type u_3
      inst✝ : (i : ι) → Monoid (N i)
      ϕ : (i : ι) → MonoidHom (N i) M
      hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
      this : DecidableEq ι := Classical.decEq ι
      ⊢ LE.le (iSup fun i => MonoidHom.mrange (ϕ i)) (MonoidHom.mrange (MonoidHom.no …
    -/
  · refine iSup_le ?_
    /-
      case a
      M : Type u_1
      inst✝² : Monoid M
      ι : Type u_2
      inst✝¹ : Fintype ι
      N : ι → Type u_3
      inst✝ : (i : ι) → Monoid (N i)
      ϕ : (i : ι) → MonoidHom (N i) M
      hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
      this : DecidableEq ι := Classical.decEq ι
      ⊢ ∀ (i : ι), LE.le (MonoidHom.mrange (ϕ i)) (MonoidHom.mrange (MonoidHom.nonco …
    -/
    rintro i x ⟨y, rfl⟩
    /-
      case a.intro
      M : Type u_1
      inst✝² : Monoid M
      ι : Type u_2
      inst✝¹ : Fintype ι
      N : ι → Type u_3
      inst✝ : (i : ι) → Monoid (N i)
      ϕ : (i : ι) → MonoidHom (N i) M
      hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
      this : DecidableEq ι := Classical.decEq ι
      i : ι
      y : N i
      ⊢ Membership.mem (MonoidHom.mrange (MonoidHom.noncommPiCoprod ϕ hcomm)) ((ϕ i) …
    -/
    exact ⟨Pi.mulSingle i y, noncommPiCoprod_mulSingle _ _ _⟩
    /-
      🎉 no goals
    -/


@[to_additive]
lemma commute_noncommPiCoprod {m : M}
    (comm : ∀ i (x : N i), Commute m ((ϕ i x))) (h : (i : ι) → N i) :
    Commute m (MonoidHom.noncommPiCoprod ϕ hcomm h) := by
  /-
    M : Type u_1
    inst✝² : Monoid M
    ι : Type u_2
    inst✝¹ : Fintype ι
    N : ι → Type u_3
    inst✝ : (i : ι) → Monoid (N i)
    ϕ : (i : ι) → MonoidHom (N i) M
    hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
    m : M
    comm : ∀ (i : ι) (x : N i), Commute m ((ϕ i) x)
    h : (i : ι) → N i
    ⊢ Commute m ((MonoidHom.noncommPiCoprod ϕ hcomm) h)
  -/
  dsimp only [MonoidHom.noncommPiCoprod, MonoidHom.coe_mk, OneHom.coe_mk]
  /-
    M : Type u_1
    inst✝² : Monoid M
    ι : Type u_2
    inst✝¹ : Fintype ι
    N : ι → Type u_3
    inst✝ : (i : ι) → Monoid (N i)
    ϕ : (i : ι) → MonoidHom (N i) M
    hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
    m : M
    comm : ∀ (i : ι) (x : N i), Commute m ((ϕ i) x)
    h : (i : ι) → N i
    ⊢ Commute m (Finset.univ.noncommProd (fun i => (ϕ i) (h i)) ⋯)
  -/
  apply Finset.noncommProd_induction
    /-
      case hom
      M : Type u_1
      inst✝² : Monoid M
      ι : Type u_2
      inst✝¹ : Fintype ι
      N : ι → Type u_3
      inst✝ : (i : ι) → Monoid (N i)
      ϕ : (i : ι) → MonoidHom (N i) M
      hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
      m : M
      comm : ∀ (i : ι) (x : N i), Commute m ((ϕ i) x)
      h : (i : ι) → N i
      ⊢ ∀ (a b : M), Commute m a → Commute m b → Commute m (HMul.hMul a b)
    -/
  · exact fun x y ↦ Commute.mul_right
    /-
      🎉 no goals
    -/
    /-
      case unit
      M : Type u_1
      inst✝² : Monoid M
      ι : Type u_2
      inst✝¹ : Fintype ι
      N : ι → Type u_3
      inst✝ : (i : ι) → Monoid (N i)
      ϕ : (i : ι) → MonoidHom (N i) M
      hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
      m : M
      comm : ∀ (i : ι) (x : N i), Commute m ((ϕ i) x)
      h : (i : ι) → N i
      ⊢ Commute m 1
    -/
  · exact Commute.one_right _
    /-
      🎉 no goals
    -/
    /-
      case base
      M : Type u_1
      inst✝² : Monoid M
      ι : Type u_2
      inst✝¹ : Fintype ι
      N : ι → Type u_3
      inst✝ : (i : ι) → Monoid (N i)
      ϕ : (i : ι) → MonoidHom (N i) M
      hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
      m : M
      comm : ∀ (i : ι) (x : N i), Commute m ((ϕ i) x)
      h : (i : ι) → N i
      ⊢ ∀ (x : ι), Membership.mem Finset.univ x → Commute m ((ϕ x) (h x))
    -/
  · exact fun x _ ↦ comm x (h x)
    /-
      🎉 no goals
    -/


@[to_additive]
lemma noncommPiCoprod_apply (h : (i : ι) → N i) :
    MonoidHom.noncommPiCoprod ϕ hcomm h = Finset.noncommProd Finset.univ (fun i ↦ ϕ i (h i))
      (Pairwise.set_pairwise (fun ⦃i j⦄ a ↦ hcomm a (h i) (h j)) _) := by
  /-
    M : Type u_1
    inst✝² : Monoid M
    ι : Type u_2
    inst✝¹ : Fintype ι
    N : ι → Type u_3
    inst✝ : (i : ι) → Monoid (N i)
    ϕ : (i : ι) → MonoidHom (N i) M
    hcomm : Pairwise fun i j => ∀ (x : N i) (y : N j), Commute ((ϕ i) x) ((ϕ j) y)
    h : (i : ι) → N i
    ⊢ Eq ((MonoidHom.noncommPiCoprod ϕ hcomm) h) (Finset.univ.noncommProd (fun i = …
  -/
  dsimp only [MonoidHom.noncommPiCoprod, MonoidHom.coe_mk, OneHom.coe_mk]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem noncommPiCoprod_range [Fintype ι]
    {hcomm : Pairwise fun i j : ι => ∀ (x : H i) (y : H j), Commute (ϕ i x) (ϕ j y)} :
    (noncommPiCoprod ϕ hcomm).range = ⨆ i : ι, (ϕ i).range := by
  /-
    G : Type u_1
    inst✝² : Group G
    ι : Type u_2
    H : ι → Type u_3
    inst✝¹ : (i : ι) → Group (H i)
    ϕ : (i : ι) → MonoidHom (H i) G
    inst✝ : Fintype ι
    hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
    ⊢ Eq (MonoidHom.noncommPiCoprod ϕ hcomm).range (iSup fun i => (ϕ i).range)
  -/
  letI := Classical.decEq ι
  /-
    G : Type u_1
    inst✝² : Group G
    ι : Type u_2
    H : ι → Type u_3
    inst✝¹ : (i : ι) → Group (H i)
    ϕ : (i : ι) → MonoidHom (H i) G
    inst✝ : Fintype ι
    hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
    this : DecidableEq ι := Classical.decEq ι
    ⊢ Eq (MonoidHom.noncommPiCoprod ϕ hcomm).range (iSup fun i => (ϕ i).range)
  -/
  apply le_antisymm
    /-
      case a
      G : Type u_1
      inst✝² : Group G
      ι : Type u_2
      H : ι → Type u_3
      inst✝¹ : (i : ι) → Group (H i)
      ϕ : (i : ι) → MonoidHom (H i) G
      inst✝ : Fintype ι
      hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
      this : DecidableEq ι := Classical.decEq ι
      ⊢ LE.le (MonoidHom.noncommPiCoprod ϕ hcomm).range (iSup fun i => (ϕ i).range)
    -/
  · rintro x ⟨f, rfl⟩
    /-
      case a.intro
      G : Type u_1
      inst✝² : Group G
      ι : Type u_2
      H : ι → Type u_3
      inst✝¹ : (i : ι) → Group (H i)
      ϕ : (i : ι) → MonoidHom (H i) G
      inst✝ : Fintype ι
      hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
      this : DecidableEq ι := Classical.decEq ι
      f : (i : ι) → H i
      ⊢ Membership.mem (iSup fun i => (ϕ i).range) ((MonoidHom.noncommPiCoprod ϕ hco …
    -/
    refine Subgroup.noncommProd_mem _ (fun _ _ _ _ h => hcomm h _ _) ?_
    /-
      case a.intro
      G : Type u_1
      inst✝² : Group G
      ι : Type u_2
      H : ι → Type u_3
      inst✝¹ : (i : ι) → Group (H i)
      ϕ : (i : ι) → MonoidHom (H i) G
      inst✝ : Fintype ι
      hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
      this : DecidableEq ι := Classical.decEq ι
      f : (i : ι) → H i
      ⊢ ∀ (c : ι), Membership.mem Finset.univ c → Membership.mem (iSup fun i => (ϕ i …
    -/
    intro i _hi
    /-
      case a.intro
      G : Type u_1
      inst✝² : Group G
      ι : Type u_2
      H : ι → Type u_3
      inst✝¹ : (i : ι) → Group (H i)
      ϕ : (i : ι) → MonoidHom (H i) G
      inst✝ : Fintype ι
      hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
      this : DecidableEq ι := Classical.decEq ι
      f : (i : ι) → H i
      i : ι
      _hi : Membership.mem Finset.univ i
      ⊢ Membership.mem (iSup fun i => (ϕ i).range) ((ϕ i) (f i))
    -/
    apply Subgroup.mem_sSup_of_mem
      /-
        case a.intro.hs
        G : Type u_1
        inst✝² : Group G
        ι : Type u_2
        H : ι → Type u_3
        inst✝¹ : (i : ι) → Group (H i)
        ϕ : (i : ι) → MonoidHom (H i) G
        inst✝ : Fintype ι
        hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
        this : DecidableEq ι := Classical.decEq ι
        f : (i : ι) → H i
        i : ι
        _hi : Membership.mem Finset.univ i
        ⊢ Membership.mem (Set.range fun i => (ϕ i).range) ?a.intro.s✝
      -/
    · use i
      /-
        🎉 no goals
      -/
    /-
      case a.intro.a
      G : Type u_1
      inst✝² : Group G
      ι : Type u_2
      H : ι → Type u_3
      inst✝¹ : (i : ι) → Group (H i)
      ϕ : (i : ι) → MonoidHom (H i) G
      inst✝ : Fintype ι
      hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
      this : DecidableEq ι := Classical.decEq ι
      f : (i : ι) → H i
      i : ι
      _hi : Membership.mem Finset.univ i
      ⊢ Membership.mem ((fun i => (ϕ i).range) i) ((ϕ i) (f i))
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case a
      G : Type u_1
      inst✝² : Group G
      ι : Type u_2
      H : ι → Type u_3
      inst✝¹ : (i : ι) → Group (H i)
      ϕ : (i : ι) → MonoidHom (H i) G
      inst✝ : Fintype ι
      hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
      this : DecidableEq ι := Classical.decEq ι
      ⊢ LE.le (iSup fun i => (ϕ i).range) (MonoidHom.noncommPiCoprod ϕ hcomm).range
    -/
  · refine iSup_le ?_
    /-
      case a
      G : Type u_1
      inst✝² : Group G
      ι : Type u_2
      H : ι → Type u_3
      inst✝¹ : (i : ι) → Group (H i)
      ϕ : (i : ι) → MonoidHom (H i) G
      inst✝ : Fintype ι
      hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
      this : DecidableEq ι := Classical.decEq ι
      ⊢ ∀ (i : ι), LE.le (ϕ i).range (MonoidHom.noncommPiCoprod ϕ hcomm).range
    -/
    rintro i x ⟨y, rfl⟩
    /-
      case a.intro
      G : Type u_1
      inst✝² : Group G
      ι : Type u_2
      H : ι → Type u_3
      inst✝¹ : (i : ι) → Group (H i)
      ϕ : (i : ι) → MonoidHom (H i) G
      inst✝ : Fintype ι
      hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
      this : DecidableEq ι := Classical.decEq ι
      i : ι
      y : H i
      ⊢ Membership.mem (MonoidHom.noncommPiCoprod ϕ hcomm).range ((ϕ i) y)
    -/
    exact ⟨Pi.mulSingle i y, noncommPiCoprod_mulSingle _ _ _⟩
    /-
      🎉 no goals
    -/


@[to_additive]
theorem injective_noncommPiCoprod_of_iSupIndep [Fintype ι]
    {hcomm : Pairwise fun i j : ι => ∀ (x : H i) (y : H j), Commute (ϕ i x) (ϕ j y)}
    (hind : iSupIndep fun i => (ϕ i).range)
    (hinj : ∀ i, Function.Injective (ϕ i)) : Function.Injective (noncommPiCoprod ϕ hcomm) := by
  classical
    apply (MonoidHom.ker_eq_bot_iff _).mp
    rw [eq_bot_iff]
    intro f heq1
    have : ∀ i, i ∈ Finset.univ → ϕ i (f i) = 1 :=
      Subgroup.eq_one_of_noncommProd_eq_one_of_iSupIndep _ _ (fun _ _ _ _ h => hcomm h _ _)
        _ hind (by simp) heq1
    ext i
    apply hinj
    simp [this i (Finset.mem_univ i)]


@[deprecated (since := "2024-11-24")]
alias injective_noncommPiCoprod_of_independent := injective_noncommPiCoprod_of_iSupIndep


@[to_additive]
theorem independent_range_of_coprime_order
    (hcomm : Pairwise fun i j : ι => ∀ (x : H i) (y : H j), Commute (ϕ i x) (ϕ j y))
    [Finite ι] [∀ i, Fintype (H i)]
    (hcoprime : Pairwise fun i j => Nat.Coprime (Fintype.card (H i)) (Fintype.card (H j))) :
    iSupIndep fun i => (ϕ i).range := by
  /-
    G : Type u_1
    inst✝³ : Group G
    ι : Type u_2
    H : ι → Type u_3
    inst✝² : (i : ι) → Group (H i)
    ϕ : (i : ι) → MonoidHom (H i) G
    hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
    inst✝¹ : Finite ι
    inst✝ : (i : ι) → Fintype (H i)
    hcoprime : Pairwise fun i j => (Fintype.card (H i)).Coprime (Fintype.card (H j))
    ⊢ iSupIndep fun i => (ϕ i).range
  -/
  cases nonempty_fintype ι
  /-
    case intro
    G : Type u_1
    inst✝³ : Group G
    ι : Type u_2
    H : ι → Type u_3
    inst✝² : (i : ι) → Group (H i)
    ϕ : (i : ι) → MonoidHom (H i) G
    hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
    inst✝¹ : Finite ι
    inst✝ : (i : ι) → Fintype (H i)
    hcoprime : Pairwise fun i j => (Fintype.card (H i)).Coprime (Fintype.card (H j))
    val✝ : Fintype ι
    ⊢ iSupIndep fun i => (ϕ i).range
  -/
  letI := Classical.decEq ι
  /-
    case intro
    G : Type u_1
    inst✝³ : Group G
    ι : Type u_2
    H : ι → Type u_3
    inst✝² : (i : ι) → Group (H i)
    ϕ : (i : ι) → MonoidHom (H i) G
    hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
    inst✝¹ : Finite ι
    inst✝ : (i : ι) → Fintype (H i)
    hcoprime : Pairwise fun i j => (Fintype.card (H i)).Coprime (Fintype.card (H j))
    val✝ : Fintype ι
    this : DecidableEq ι := Classical.decEq ι
    ⊢ iSupIndep fun i => (ϕ i).range
  -/
  rintro i
  /-
    case intro
    G : Type u_1
    inst✝³ : Group G
    ι : Type u_2
    H : ι → Type u_3
    inst✝² : (i : ι) → Group (H i)
    ϕ : (i : ι) → MonoidHom (H i) G
    hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
    inst✝¹ : Finite ι
    inst✝ : (i : ι) → Fintype (H i)
    hcoprime : Pairwise fun i j => (Fintype.card (H i)).Coprime (Fintype.card (H j))
    val✝ : Fintype ι
    this : DecidableEq ι := Classical.decEq ι
    i : ι
    ⊢ Disjoint ((fun i => (ϕ i).range) i) (iSup fun j => iSup fun x => (fun i => ( …
  -/
  rw [disjoint_iff_inf_le]
  /-
    case intro
    G : Type u_1
    inst✝³ : Group G
    ι : Type u_2
    H : ι → Type u_3
    inst✝² : (i : ι) → Group (H i)
    ϕ : (i : ι) → MonoidHom (H i) G
    hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
    inst✝¹ : Finite ι
    inst✝ : (i : ι) → Fintype (H i)
    hcoprime : Pairwise fun i j => (Fintype.card (H i)).Coprime (Fintype.card (H j))
    val✝ : Fintype ι
    this : DecidableEq ι := Classical.decEq ι
    i : ι
    ⊢ LE.le (Min.min ((fun i => (ϕ i).range) i) (iSup fun j => iSup fun x => (fun  …
  -/
  rintro f ⟨hxi, hxp⟩
  /-
    case intro.intro
    G : Type u_1
    inst✝³ : Group G
    ι : Type u_2
    H : ι → Type u_3
    inst✝² : (i : ι) → Group (H i)
    ϕ : (i : ι) → MonoidHom (H i) G
    hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
    inst✝¹ : Finite ι
    inst✝ : (i : ι) → Fintype (H i)
    hcoprime : Pairwise fun i j => (Fintype.card (H i)).Coprime (Fintype.card (H j))
    val✝ : Fintype ι
    this : DecidableEq ι := Classical.decEq ι
    i : ι
    f : G
    hxi : Membership.mem (↑((fun i => (ϕ i).range) i).toSubmonoid) f
    hxp : Membership.mem (↑(iSup fun j => iSup fun x => (fun i => (ϕ i).range) j). …
    ⊢ Membership.mem Bot.bot f
  -/
  dsimp at hxi hxp
  /-
    case intro.intro
    G : Type u_1
    inst✝³ : Group G
    ι : Type u_2
    H : ι → Type u_3
    inst✝² : (i : ι) → Group (H i)
    ϕ : (i : ι) → MonoidHom (H i) G
    hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
    inst✝¹ : Finite ι
    inst✝ : (i : ι) → Fintype (H i)
    hcoprime : Pairwise fun i j => (Fintype.card (H i)).Coprime (Fintype.card (H j))
    val✝ : Fintype ι
    this : DecidableEq ι := Classical.decEq ι
    i : ι
    f : G
    hxi : Membership.mem (Set.range ⇑(ϕ i)) f
    hxp : Membership.mem (↑(iSup fun j => iSup fun x => (ϕ j).range)) f
    ⊢ Membership.mem Bot.bot f
  -/
  rw [iSup_subtype', ← noncommPiCoprod_range] at hxp
  /-
    case intro.intro
    G : Type u_1
    inst✝³ : Group G
    ι : Type u_2
    H : ι → Type u_3
    inst✝² : (i : ι) → Group (H i)
    ϕ : (i : ι) → MonoidHom (H i) G
    hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
    inst✝¹ : Finite ι
    inst✝ : (i : ι) → Fintype (H i)
    hcoprime : Pairwise fun i j => (Fintype.card (H i)).Coprime (Fintype.card (H j))
    val✝ : Fintype ι
    this : DecidableEq ι := Classical.decEq ι
    i : ι
    f : G
    hxi : Membership.mem (Set.range ⇑(ϕ i)) f
    hxp✝ : Membership.mem (↑(iSup fun x => (ϕ ↑x).range)) f
    hxp : Membership.mem (↑(MonoidHom.noncommPiCoprod (fun x => ϕ ↑x) ?intro.intro …
    ⊢ Membership.mem Bot.bot f
  -/
  rotate_left
    /-
      case intro.intro.hcomm
      G : Type u_1
      inst✝³ : Group G
      ι : Type u_2
      H : ι → Type u_3
      inst✝² : (i : ι) → Group (H i)
      ϕ : (i : ι) → MonoidHom (H i) G
      hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
      inst✝¹ : Finite ι
      inst✝ : (i : ι) → Fintype (H i)
      hcoprime : Pairwise fun i j => (Fintype.card (H i)).Coprime (Fintype.card (H j))
      val✝ : Fintype ι
      this : DecidableEq ι := Classical.decEq ι
      i : ι
      f : G
      hxi : Membership.mem (Set.range ⇑(ϕ i)) f
      hxp : Membership.mem (↑(iSup fun x => (ϕ ↑x).range)) f
      ⊢ Pairwise fun i_1 j => ∀ (x : H ↑i_1) (y : H ↑j), Commute ((ϕ ↑i_1) x) ((ϕ ↑j …
    -/
  · intro _ _ hj
    /-
      case intro.intro.hcomm
      G : Type u_1
      inst✝³ : Group G
      ι : Type u_2
      H : ι → Type u_3
      inst✝² : (i : ι) → Group (H i)
      ϕ : (i : ι) → MonoidHom (H i) G
      hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
      inst✝¹ : Finite ι
      inst✝ : (i : ι) → Fintype (H i)
      hcoprime : Pairwise fun i j => (Fintype.card (H i)).Coprime (Fintype.card (H j))
      val✝ : Fintype ι
      this : DecidableEq ι := Classical.decEq ι
      i : ι
      f : G
      hxi : Membership.mem (Set.range ⇑(ϕ i)) f
      hxp : Membership.mem (↑(iSup fun x => (ϕ ↑x).range)) f
      i✝ j✝ : Subtype fun j => Not (Eq j i)
      hj : Ne i✝ j✝
      ⊢ ∀ (x : H ↑i✝) (y : H ↑j✝), Commute ((ϕ ↑i✝) x) ((ϕ ↑j✝) y)
    -/
    apply hcomm
    /-
      case intro.intro.hcomm.a
      G : Type u_1
      inst✝³ : Group G
      ι : Type u_2
      H : ι → Type u_3
      inst✝² : (i : ι) → Group (H i)
      ϕ : (i : ι) → MonoidHom (H i) G
      hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
      inst✝¹ : Finite ι
      inst✝ : (i : ι) → Fintype (H i)
      hcoprime : Pairwise fun i j => (Fintype.card (H i)).Coprime (Fintype.card (H j))
      val✝ : Fintype ι
      this : DecidableEq ι := Classical.decEq ι
      i : ι
      f : G
      hxi : Membership.mem (Set.range ⇑(ϕ i)) f
      hxp : Membership.mem (↑(iSup fun x => (ϕ ↑x).range)) f
      i✝ j✝ : Subtype fun j => Not (Eq j i)
      hj : Ne i✝ j✝
      ⊢ Ne ↑i✝ ↑j✝
    -/
    exact hj ∘ Subtype.ext
    /-
      🎉 no goals
    -/
  /-
    case intro.intro
    G : Type u_1
    inst✝³ : Group G
    ι : Type u_2
    H : ι → Type u_3
    inst✝² : (i : ι) → Group (H i)
    ϕ : (i : ι) → MonoidHom (H i) G
    hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
    inst✝¹ : Finite ι
    inst✝ : (i : ι) → Fintype (H i)
    hcoprime : Pairwise fun i j => (Fintype.card (H i)).Coprime (Fintype.card (H j))
    val✝ : Fintype ι
    this : DecidableEq ι := Classical.decEq ι
    i : ι
    f : G
    hxi : Membership.mem (Set.range ⇑(ϕ i)) f
    hxp✝ : Membership.mem (↑(iSup fun x => (ϕ ↑x).range)) f
    hxp : Membership.mem (↑(MonoidHom.noncommPiCoprod (fun x => ϕ ↑x) ⋯).range) f
    ⊢ Membership.mem Bot.bot f
  -/
  cases' hxp with g hgf
  /-
    case intro.intro.intro
    G : Type u_1
    inst✝³ : Group G
    ι : Type u_2
    H : ι → Type u_3
    inst✝² : (i : ι) → Group (H i)
    ϕ : (i : ι) → MonoidHom (H i) G
    hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
    inst✝¹ : Finite ι
    inst✝ : (i : ι) → Fintype (H i)
    hcoprime : Pairwise fun i j => (Fintype.card (H i)).Coprime (Fintype.card (H j))
    val✝ : Fintype ι
    this : DecidableEq ι := Classical.decEq ι
    i : ι
    f : G
    hxi : Membership.mem (Set.range ⇑(ϕ i)) f
    hxp : Membership.mem (↑(iSup fun x => (ϕ ↑x).range)) f
    g : (i_1 : Subtype fun j => Not (Eq j i)) → H ↑i_1
    hgf : Eq ((MonoidHom.noncommPiCoprod (fun x => ϕ ↑x) ⋯) g) f
    ⊢ Membership.mem Bot.bot f
  -/
  cases' hxi with g' hg'f
  have hxi : orderOf f ∣ Fintype.card (H i) := by
    rw [← hg'f]
    exact (orderOf_map_dvd _ _).trans orderOf_dvd_card
  have hxp : orderOf f ∣ ∏ j : { j // j ≠ i }, Fintype.card (H j) := by
    rw [← hgf, ← Fintype.card_pi]
    exact (orderOf_map_dvd _ _).trans orderOf_dvd_card
  /-
    case intro.intro.intro.intro
    G : Type u_1
    inst✝³ : Group G
    ι : Type u_2
    H : ι → Type u_3
    inst✝² : (i : ι) → Group (H i)
    ϕ : (i : ι) → MonoidHom (H i) G
    hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
    inst✝¹ : Finite ι
    inst✝ : (i : ι) → Fintype (H i)
    hcoprime : Pairwise fun i j => (Fintype.card (H i)).Coprime (Fintype.card (H j))
    val✝ : Fintype ι
    this : DecidableEq ι := Classical.decEq ι
    i : ι
    f : G
    hxp✝ : Membership.mem (↑(iSup fun x => (ϕ ↑x).range)) f
    g : (i_1 : Subtype fun j => Not (Eq j i)) → H ↑i_1
    hgf : Eq ((MonoidHom.noncommPiCoprod (fun x => ϕ ↑x) ⋯) g) f
    g' : H i
    hg'f : Eq ((ϕ i) g') f
    hxi : Dvd.dvd (orderOf f) (Fintype.card (H i))
    hxp : Dvd.dvd (orderOf f) (Finset.univ.prod fun j => Fintype.card (H ↑j))
    ⊢ Membership.mem Bot.bot f
  -/
  change f = 1
  /-
    case intro.intro.intro.intro
    G : Type u_1
    inst✝³ : Group G
    ι : Type u_2
    H : ι → Type u_3
    inst✝² : (i : ι) → Group (H i)
    ϕ : (i : ι) → MonoidHom (H i) G
    hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
    inst✝¹ : Finite ι
    inst✝ : (i : ι) → Fintype (H i)
    hcoprime : Pairwise fun i j => (Fintype.card (H i)).Coprime (Fintype.card (H j))
    val✝ : Fintype ι
    this : DecidableEq ι := Classical.decEq ι
    i : ι
    f : G
    hxp✝ : Membership.mem (↑(iSup fun x => (ϕ ↑x).range)) f
    g : (i_1 : Subtype fun j => Not (Eq j i)) → H ↑i_1
    hgf : Eq ((MonoidHom.noncommPiCoprod (fun x => ϕ ↑x) ⋯) g) f
    g' : H i
    hg'f : Eq ((ϕ i) g') f
    hxi : Dvd.dvd (orderOf f) (Fintype.card (H i))
    hxp : Dvd.dvd (orderOf f) (Finset.univ.prod fun j => Fintype.card (H ↑j))
    ⊢ Eq f 1
  -/
  rw [← pow_one f, ← orderOf_dvd_iff_pow_eq_one]
  -- Porting note: ouch, had to replace an ugly `convert`
  /-
    case intro.intro.intro.intro
    G : Type u_1
    inst✝³ : Group G
    ι : Type u_2
    H : ι → Type u_3
    inst✝² : (i : ι) → Group (H i)
    ϕ : (i : ι) → MonoidHom (H i) G
    hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
    inst✝¹ : Finite ι
    inst✝ : (i : ι) → Fintype (H i)
    hcoprime : Pairwise fun i j => (Fintype.card (H i)).Coprime (Fintype.card (H j))
    val✝ : Fintype ι
    this : DecidableEq ι := Classical.decEq ι
    i : ι
    f : G
    hxp✝ : Membership.mem (↑(iSup fun x => (ϕ ↑x).range)) f
    g : (i_1 : Subtype fun j => Not (Eq j i)) → H ↑i_1
    hgf : Eq ((MonoidHom.noncommPiCoprod (fun x => ϕ ↑x) ⋯) g) f
    g' : H i
    hg'f : Eq ((ϕ i) g') f
    hxi : Dvd.dvd (orderOf f) (Fintype.card (H i))
    hxp : Dvd.dvd (orderOf f) (Finset.univ.prod fun j => Fintype.card (H ↑j))
    ⊢ Dvd.dvd (orderOf f) 1
  -/
  obtain ⟨c, hc⟩ := Nat.dvd_gcd hxp hxi
  /-
    case intro.intro.intro.intro.intro
    G : Type u_1
    inst✝³ : Group G
    ι : Type u_2
    H : ι → Type u_3
    inst✝² : (i : ι) → Group (H i)
    ϕ : (i : ι) → MonoidHom (H i) G
    hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
    inst✝¹ : Finite ι
    inst✝ : (i : ι) → Fintype (H i)
    hcoprime : Pairwise fun i j => (Fintype.card (H i)).Coprime (Fintype.card (H j))
    val✝ : Fintype ι
    this : DecidableEq ι := Classical.decEq ι
    i : ι
    f : G
    hxp✝ : Membership.mem (↑(iSup fun x => (ϕ ↑x).range)) f
    g : (i_1 : Subtype fun j => Not (Eq j i)) → H ↑i_1
    hgf : Eq ((MonoidHom.noncommPiCoprod (fun x => ϕ ↑x) ⋯) g) f
    g' : H i
    hg'f : Eq ((ϕ i) g') f
    hxi : Dvd.dvd (orderOf f) (Fintype.card (H i))
    hxp : Dvd.dvd (orderOf f) (Finset.univ.prod fun j => Fintype.card (H ↑j))
    c : Nat
    hc : Eq ((Finset.univ.prod fun j => Fintype.card (H ↑j)).gcd (Fintype.card (H  …
    ⊢ Dvd.dvd (orderOf f) 1
  -/
  use c
  /-
    case h
    G : Type u_1
    inst✝³ : Group G
    ι : Type u_2
    H : ι → Type u_3
    inst✝² : (i : ι) → Group (H i)
    ϕ : (i : ι) → MonoidHom (H i) G
    hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
    inst✝¹ : Finite ι
    inst✝ : (i : ι) → Fintype (H i)
    hcoprime : Pairwise fun i j => (Fintype.card (H i)).Coprime (Fintype.card (H j))
    val✝ : Fintype ι
    this : DecidableEq ι := Classical.decEq ι
    i : ι
    f : G
    hxp✝ : Membership.mem (↑(iSup fun x => (ϕ ↑x).range)) f
    g : (i_1 : Subtype fun j => Not (Eq j i)) → H ↑i_1
    hgf : Eq ((MonoidHom.noncommPiCoprod (fun x => ϕ ↑x) ⋯) g) f
    g' : H i
    hg'f : Eq ((ϕ i) g') f
    hxi : Dvd.dvd (orderOf f) (Fintype.card (H i))
    hxp : Dvd.dvd (orderOf f) (Finset.univ.prod fun j => Fintype.card (H ↑j))
    c : Nat
    hc : Eq ((Finset.univ.prod fun j => Fintype.card (H ↑j)).gcd (Fintype.card (H  …
    ⊢ Eq 1 (HMul.hMul (orderOf f) c)
  -/
  rw [← hc]
  /-
    case h
    G : Type u_1
    inst✝³ : Group G
    ι : Type u_2
    H : ι → Type u_3
    inst✝² : (i : ι) → Group (H i)
    ϕ : (i : ι) → MonoidHom (H i) G
    hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
    inst✝¹ : Finite ι
    inst✝ : (i : ι) → Fintype (H i)
    hcoprime : Pairwise fun i j => (Fintype.card (H i)).Coprime (Fintype.card (H j))
    val✝ : Fintype ι
    this : DecidableEq ι := Classical.decEq ι
    i : ι
    f : G
    hxp✝ : Membership.mem (↑(iSup fun x => (ϕ ↑x).range)) f
    g : (i_1 : Subtype fun j => Not (Eq j i)) → H ↑i_1
    hgf : Eq ((MonoidHom.noncommPiCoprod (fun x => ϕ ↑x) ⋯) g) f
    g' : H i
    hg'f : Eq ((ϕ i) g') f
    hxi : Dvd.dvd (orderOf f) (Fintype.card (H i))
    hxp : Dvd.dvd (orderOf f) (Finset.univ.prod fun j => Fintype.card (H ↑j))
    c : Nat
    hc : Eq ((Finset.univ.prod fun j => Fintype.card (H ↑j)).gcd (Fintype.card (H  …
    ⊢ Eq 1 ((Finset.univ.prod fun j => Fintype.card (H ↑j)).gcd (Fintype.card (H i …
  -/
  symm
  /-
    case h
    G : Type u_1
    inst✝³ : Group G
    ι : Type u_2
    H : ι → Type u_3
    inst✝² : (i : ι) → Group (H i)
    ϕ : (i : ι) → MonoidHom (H i) G
    hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
    inst✝¹ : Finite ι
    inst✝ : (i : ι) → Fintype (H i)
    hcoprime : Pairwise fun i j => (Fintype.card (H i)).Coprime (Fintype.card (H j))
    val✝ : Fintype ι
    this : DecidableEq ι := Classical.decEq ι
    i : ι
    f : G
    hxp✝ : Membership.mem (↑(iSup fun x => (ϕ ↑x).range)) f
    g : (i_1 : Subtype fun j => Not (Eq j i)) → H ↑i_1
    hgf : Eq ((MonoidHom.noncommPiCoprod (fun x => ϕ ↑x) ⋯) g) f
    g' : H i
    hg'f : Eq ((ϕ i) g') f
    hxi : Dvd.dvd (orderOf f) (Fintype.card (H i))
    hxp : Dvd.dvd (orderOf f) (Finset.univ.prod fun j => Fintype.card (H ↑j))
    c : Nat
    hc : Eq ((Finset.univ.prod fun j => Fintype.card (H ↑j)).gcd (Fintype.card (H  …
    ⊢ Eq ((Finset.univ.prod fun j => Fintype.card (H ↑j)).gcd (Fintype.card (H i)) …
  -/
  rw [← Nat.coprime_iff_gcd_eq_one, Nat.coprime_fintype_prod_left_iff, Subtype.forall]
  /-
    case h
    G : Type u_1
    inst✝³ : Group G
    ι : Type u_2
    H : ι → Type u_3
    inst✝² : (i : ι) → Group (H i)
    ϕ : (i : ι) → MonoidHom (H i) G
    hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
    inst✝¹ : Finite ι
    inst✝ : (i : ι) → Fintype (H i)
    hcoprime : Pairwise fun i j => (Fintype.card (H i)).Coprime (Fintype.card (H j))
    val✝ : Fintype ι
    this : DecidableEq ι := Classical.decEq ι
    i : ι
    f : G
    hxp✝ : Membership.mem (↑(iSup fun x => (ϕ ↑x).range)) f
    g : (i_1 : Subtype fun j => Not (Eq j i)) → H ↑i_1
    hgf : Eq ((MonoidHom.noncommPiCoprod (fun x => ϕ ↑x) ⋯) g) f
    g' : H i
    hg'f : Eq ((ϕ i) g') f
    hxi : Dvd.dvd (orderOf f) (Fintype.card (H i))
    hxp : Dvd.dvd (orderOf f) (Finset.univ.prod fun j => Fintype.card (H ↑j))
    c : Nat
    hc : Eq ((Finset.univ.prod fun j => Fintype.card (H ↑j)).gcd (Fintype.card (H  …
    ⊢ ∀ (a : ι) (b : Ne a i), (Fintype.card (H ↑⟨a, b⟩)).Coprime (Fintype.card (H  …
  -/
  intro j h
  /-
    case h
    G : Type u_1
    inst✝³ : Group G
    ι : Type u_2
    H : ι → Type u_3
    inst✝² : (i : ι) → Group (H i)
    ϕ : (i : ι) → MonoidHom (H i) G
    hcomm : Pairwise fun i j => ∀ (x : H i) (y : H j), Commute ((ϕ i) x) ((ϕ j) y)
    inst✝¹ : Finite ι
    inst✝ : (i : ι) → Fintype (H i)
    hcoprime : Pairwise fun i j => (Fintype.card (H i)).Coprime (Fintype.card (H j))
    val✝ : Fintype ι
    this : DecidableEq ι := Classical.decEq ι
    i : ι
    f : G
    hxp✝ : Membership.mem (↑(iSup fun x => (ϕ ↑x).range)) f
    g : (i_1 : Subtype fun j => Not (Eq j i)) → H ↑i_1
    hgf : Eq ((MonoidHom.noncommPiCoprod (fun x => ϕ ↑x) ⋯) g) f
    g' : H i
    hg'f : Eq ((ϕ i) g') f
    hxi : Dvd.dvd (orderOf f) (Fintype.card (H i))
    hxp : Dvd.dvd (orderOf f) (Finset.univ.prod fun j => Fintype.card (H ↑j))
    c : Nat
    hc : Eq ((Finset.univ.prod fun j => Fintype.card (H ↑j)).gcd (Fintype.card (H  …
    j : ι
    h : Ne j i
    ⊢ (Fintype.card (H ↑⟨j, h⟩)).Coprime (Fintype.card (H i))
  -/
  exact hcoprime h
  /-
    🎉 no goals
  -/


@[to_additive]
theorem commute_subtype_of_commute
    (hcomm : Pairwise fun i j : ι => ∀ x y : G, x ∈ H i → y ∈ H j → Commute x y) (i j : ι)
    (hne : i ≠ j) :
    ∀ (x : H i) (y : H j), Commute ((H i).subtype x) ((H j).subtype y) := by
  /-
    G : Type u_1
    inst✝ : Group G
    ι : Type u_2
    H : ι → Subgroup G
    hcomm : Pairwise fun i j => ∀ (x y : G), Membership.mem (H i) x → Membership.m …
    i j : ι
    hne : Ne i j
    ⊢ ∀ (x : Subtype fun x => Membership.mem (H i) x) (y : Subtype fun x => Member …
  -/
  rintro ⟨x, hx⟩ ⟨y, hy⟩
  /-
    case mk.mk
    G : Type u_1
    inst✝ : Group G
    ι : Type u_2
    H : ι → Subgroup G
    hcomm : Pairwise fun i j => ∀ (x y : G), Membership.mem (H i) x → Membership.m …
    i j : ι
    hne : Ne i j
    x : G
    hx : Membership.mem (H i) x
    y : G
    hy : Membership.mem (H j) y
    ⊢ Commute ((H i).subtype ⟨x, hx⟩) ((H j).subtype ⟨y, hy⟩)
  -/
  exact hcomm hne x y hx hy
  /-
    🎉 no goals
  -/


@[to_additive]
theorem independent_of_coprime_order
    (hcomm : Pairwise fun i j : ι => ∀ x y : G, x ∈ H i → y ∈ H j → Commute x y)
    [Finite ι] [∀ i, Fintype (H i)]
    (hcoprime : Pairwise fun i j => Nat.Coprime (Fintype.card (H i)) (Fintype.card (H j))) :
    iSupIndep H := by
  simpa using
    MonoidHom.independent_range_of_coprime_order (fun i => (H i).subtype)
      (commute_subtype_of_commute hcomm) hcoprime


/-- The canonical homomorphism from a family of subgroups where elements from different subgroups
commute -/
@[to_additive "The canonical homomorphism from a family of additive subgroups where elements from
different subgroups commute"]
def noncommPiCoprod (hcomm : Pairwise fun i j : ι => ∀ x y : G, x ∈ H i → y ∈ H j → Commute x y) :
    (∀ i : ι, H i) →* G :=
  MonoidHom.noncommPiCoprod (fun i => (H i).subtype) (commute_subtype_of_commute hcomm)


@[to_additive (attr := simp)]
theorem noncommPiCoprod_mulSingle [DecidableEq ι]
    {hcomm : Pairwise fun i j : ι => ∀ x y : G, x ∈ H i → y ∈ H j → Commute x y}(i : ι) (y : H i) :
                                                       /-
                                                         G : Type u_1
                                                         inst✝² : Group G
                                                         ι : Type u_2
                                                         H : ι → Subgroup G
                                                         inst✝¹ : Fintype ι
                                                         inst✝ : DecidableEq ι
                                                         hcomm : Pairwise fun i j => ∀ (x y : G), Membership.mem (H i) x → Membership.m …
                                                         i : ι
                                                         y : Subtype fun x => Membership.mem (H i) x
                                                         ⊢ Eq ((Subgroup.noncommPiCoprod hcomm) (Pi.mulSingle i y)) ↑y
                                                       -/
    noncommPiCoprod hcomm (Pi.mulSingle i y) = y := by apply MonoidHom.noncommPiCoprod_mulSingle
                                                       /-
                                                         🎉 no goals
                                                       -/


@[to_additive]
theorem noncommPiCoprod_range
    {hcomm : Pairwise fun i j : ι => ∀ x y : G, x ∈ H i → y ∈ H j → Commute x y} :
    (noncommPiCoprod hcomm).range = ⨆ i : ι, H i := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    ι : Type u_2
    H : ι → Subgroup G
    inst✝ : Fintype ι
    hcomm : Pairwise fun i j => ∀ (x y : G), Membership.mem (H i) x → Membership.m …
    ⊢ Eq (Subgroup.noncommPiCoprod hcomm).range (iSup fun i => H i)
  -/
  simp [noncommPiCoprod, MonoidHom.noncommPiCoprod_range]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem injective_noncommPiCoprod_of_iSupIndep
    {hcomm : Pairwise fun i j : ι => ∀ x y : G, x ∈ H i → y ∈ H j → Commute x y}
    (hind : iSupIndep H) :
    Function.Injective (noncommPiCoprod hcomm) := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    ι : Type u_2
    H : ι → Subgroup G
    inst✝ : Fintype ι
    hcomm : Pairwise fun i j => ∀ (x y : G), Membership.mem (H i) x → Membership.m …
    hind : iSupIndep H
    ⊢ Function.Injective ⇑(Subgroup.noncommPiCoprod hcomm)
  -/
  apply MonoidHom.injective_noncommPiCoprod_of_iSupIndep
    /-
      case hind
      G : Type u_1
      inst✝¹ : Group G
      ι : Type u_2
      H : ι → Subgroup G
      inst✝ : Fintype ι
      hcomm : Pairwise fun i j => ∀ (x y : G), Membership.mem (H i) x → Membership.m …
      hind : iSupIndep H
      ⊢ iSupIndep fun i => (H i).subtype.range
    -/
  · simpa using hind
    /-
      🎉 no goals
    -/
    /-
      case hinj
      G : Type u_1
      inst✝¹ : Group G
      ι : Type u_2
      H : ι → Subgroup G
      inst✝ : Fintype ι
      hcomm : Pairwise fun i j => ∀ (x y : G), Membership.mem (H i) x → Membership.m …
      hind : iSupIndep H
      ⊢ ∀ (i : ι), Function.Injective ⇑(H i).subtype
    -/
  · intro i
    /-
      case hinj
      G : Type u_1
      inst✝¹ : Group G
      ι : Type u_2
      H : ι → Subgroup G
      inst✝ : Fintype ι
      hcomm : Pairwise fun i j => ∀ (x y : G), Membership.mem (H i) x → Membership.m …
      hind : iSupIndep H
      i : ι
      ⊢ Function.Injective ⇑(H i).subtype
    -/
    exact Subtype.coe_injective
    /-
      🎉 no goals
    -/


@[to_additive]
theorem noncommPiCoprod_apply (comm) (u : (i : ι) → H i) :
    Subgroup.noncommPiCoprod comm u = Finset.noncommProd Finset.univ (fun i ↦ u i)
      (fun i _ j _ h ↦ comm h _ _ (u i).prop (u j).prop) := by
  simp only [Subgroup.noncommPiCoprod, MonoidHom.noncommPiCoprod,
    coeSubtype, MonoidHom.coe_mk, OneHom.coe_mk]


