/-- The finset of functions `ι →₀ μ` with support contained in `s` and sum equal to `n`. -/
def finsuppAntidiag (s : Finset ι) (n : μ) : Finset (ι →₀ μ) :=
  (piAntidiag s n).attach.map ⟨fun f ↦ ⟨s.filter (f.1 · ≠ 0), f.1, by
    /-
      ι : Type u_1
      μ : Type u_2
      μ' : Type u_3
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      s✝ : Finset ι
      n✝ : μ
      f✝ : Finsupp ι μ
      s : Finset ι
      n : μ
      f : Subtype fun x => Membership.mem (s.piAntidiag n) x
      ⊢ ∀ (a : ι), Iff (Membership.mem (Finset.filter (fun x => Ne (↑f x) 0) s) a) ( …
    -/
    simpa using (mem_piAntidiag.1 f.2).2⟩, fun _ _ hfg ↦ Subtype.ext (congr_arg (⇑) hfg)⟩
    /-
      🎉 no goals
    -/


@[simp] lemma mem_finsuppAntidiag : f ∈ finsuppAntidiag s n ↔ s.sum f = n ∧ f.support ⊆ s := by
  /-
    ι : Type u_1
    μ : Type u_2
    inst✝³ : DecidableEq ι
    inst✝² : AddCommMonoid μ
    inst✝¹ : Finset.HasAntidiagonal μ
    inst✝ : DecidableEq μ
    s : Finset ι
    n : μ
    f : Finsupp ι μ
    ⊢ Iff (Membership.mem (s.finsuppAntidiag n) f) (And (Eq (s.sum ⇑f) n) (HasSubs …
  -/
  simp [finsuppAntidiag, ← DFunLike.coe_fn_eq, subset_iff]
  /-
    🎉 no goals
  -/


lemma mem_finsuppAntidiag' :
    f ∈ finsuppAntidiag s n ↔ f.sum (fun _ x ↦ x) = n ∧ f.support ⊆ s := by
  /-
    ι : Type u_1
    μ : Type u_2
    inst✝³ : DecidableEq ι
    inst✝² : AddCommMonoid μ
    inst✝¹ : Finset.HasAntidiagonal μ
    inst✝ : DecidableEq μ
    s : Finset ι
    n : μ
    f : Finsupp ι μ
    ⊢ Iff (Membership.mem (s.finsuppAntidiag n) f) (And (Eq (f.sum fun x x => x) n …
  -/
  simp only [mem_finsuppAntidiag, and_congr_left_iff]
  /-
    ι : Type u_1
    μ : Type u_2
    inst✝³ : DecidableEq ι
    inst✝² : AddCommMonoid μ
    inst✝¹ : Finset.HasAntidiagonal μ
    inst✝ : DecidableEq μ
    s : Finset ι
    n : μ
    f : Finsupp ι μ
    ⊢ HasSubset.Subset f.support s → Iff (Eq (s.sum ⇑f) n) (Eq (f.sum fun x x => x …
  -/
  rintro hf
  /-
    ι : Type u_1
    μ : Type u_2
    inst✝³ : DecidableEq ι
    inst✝² : AddCommMonoid μ
    inst✝¹ : Finset.HasAntidiagonal μ
    inst✝ : DecidableEq μ
    s : Finset ι
    n : μ
    f : Finsupp ι μ
    hf : HasSubset.Subset f.support s
    ⊢ Iff (Eq (s.sum ⇑f) n) (Eq (f.sum fun x x => x) n)
  -/
  rw [sum_of_support_subset (N := μ) f hf (fun _ x ↦ x) fun _ _ ↦ rfl]
  /-
    🎉 no goals
  -/


@[simp] lemma finsuppAntidiag_empty_zero : finsuppAntidiag (∅ : Finset ι) (0 : μ) = {0} := by
  /-
    ι : Type u_1
    μ : Type u_2
    inst✝³ : DecidableEq ι
    inst✝² : AddCommMonoid μ
    inst✝¹ : Finset.HasAntidiagonal μ
    inst✝ : DecidableEq μ
    ⊢ Eq (EmptyCollection.emptyCollection.finsuppAntidiag 0) (Singleton.singleton 0)
  -/
  ext f; simp [finsuppAntidiag, ← DFunLike.coe_fn_eq (g := f), eq_comm]
         /-
           🎉 no goals
         -/


@[simp] lemma finsuppAntidiag_empty_of_ne_zero (hn : n ≠ 0) :
    finsuppAntidiag (∅ : Finset ι) n = ∅ :=
                                 /-
                                   ι : Type u_1
                                   μ : Type u_2
                                   inst✝³ : DecidableEq ι
                                   inst✝² : AddCommMonoid μ
                                   inst✝¹ : Finset.HasAntidiagonal μ
                                   inst✝ : DecidableEq μ
                                   n : μ
                                   hn : Ne n 0
                                   ⊢ ∀ (x : Finsupp ι μ), Not (Membership.mem (EmptyCollection.emptyCollection.fi …
                                 -/
  eq_empty_of_forall_not_mem (by simp [@eq_comm _ 0, hn.symm])
                                 /-
                                   🎉 no goals
                                 -/


lemma finsuppAntidiag_empty (n : μ) :
                                                                      /-
                                                                        ι : Type u_1
                                                                        μ : Type u_2
                                                                        inst✝³ : DecidableEq ι
                                                                        inst✝² : AddCommMonoid μ
                                                                        inst✝¹ : Finset.HasAntidiagonal μ
                                                                        inst✝ : DecidableEq μ
                                                                        n : μ
                                                                        ⊢ Eq (EmptyCollection.emptyCollection.finsuppAntidiag n) (ite (Eq n 0) (Single …
                                                                      -/
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/
    finsuppAntidiag (∅ : Finset ι) n = if n = 0 then {0} else ∅ := by split_ifs with hn <;> simp [*]
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


theorem mem_finsuppAntidiag_insert {a : ι} {s : Finset ι}
    (h : a ∉ s) (n : μ) {f : ι →₀ μ} :
    f ∈ finsuppAntidiag (insert a s) n ↔
      ∃ m ∈ antidiagonal n, ∃ (g : ι →₀ μ),
        f = Finsupp.update g a m.1 ∧ g ∈ finsuppAntidiag s m.2 := by
  /-
    ι : Type u_1
    μ : Type u_2
    inst✝³ : DecidableEq ι
    inst✝² : AddCommMonoid μ
    inst✝¹ : Finset.HasAntidiagonal μ
    inst✝ : DecidableEq μ
    a : ι
    s : Finset ι
    h : Not (Membership.mem s a)
    n : μ
    f : Finsupp ι μ
    ⊢ Iff (Membership.mem ((Insert.insert a s).finsuppAntidiag n) f) (Exists fun m …
  -/
  simp only [mem_finsuppAntidiag, mem_antidiagonal, Prod.exists, sum_insert h]
  /-
    ι : Type u_1
    μ : Type u_2
    inst✝³ : DecidableEq ι
    inst✝² : AddCommMonoid μ
    inst✝¹ : Finset.HasAntidiagonal μ
    inst✝ : DecidableEq μ
    a : ι
    s : Finset ι
    h : Not (Membership.mem s a)
    n : μ
    f : Finsupp ι μ
    ⊢ Iff (And (Eq (HAdd.hAdd (f a) (s.sum fun x => f x)) n) (HasSubset.Subset f.s …
  -/
  constructor
    /-
      case mp
      ι : Type u_1
      μ : Type u_2
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      a : ι
      s : Finset ι
      h : Not (Membership.mem s a)
      n : μ
      f : Finsupp ι μ
      ⊢ And (Eq (HAdd.hAdd (f a) (s.sum fun x => f x)) n) (HasSubset.Subset f.suppor …
    -/
  · rintro ⟨rfl, hsupp⟩
    /-
      case mp.intro
      ι : Type u_1
      μ : Type u_2
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      a : ι
      s : Finset ι
      h : Not (Membership.mem s a)
      f : Finsupp ι μ
      hsupp : HasSubset.Subset f.support (Insert.insert a s)
      ⊢ Exists fun a_1 => Exists fun b => And (Eq (HAdd.hAdd a_1 b) (HAdd.hAdd (f a) …
    -/
    refine ⟨_, _, rfl, Finsupp.erase a f, ?_, ?_, ?_⟩
      /-
        case mp.intro.refine_1
        ι : Type u_1
        μ : Type u_2
        inst✝³ : DecidableEq ι
        inst✝² : AddCommMonoid μ
        inst✝¹ : Finset.HasAntidiagonal μ
        inst✝ : DecidableEq μ
        a : ι
        s : Finset ι
        h : Not (Membership.mem s a)
        f : Finsupp ι μ
        hsupp : HasSubset.Subset f.support (Insert.insert a s)
        ⊢ Eq f ((Finsupp.erase a f).update a (f a))
      -/
    · rw [update_erase_eq_update, Finsupp.update_self]
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.refine_2
        ι : Type u_1
        μ : Type u_2
        inst✝³ : DecidableEq ι
        inst✝² : AddCommMonoid μ
        inst✝¹ : Finset.HasAntidiagonal μ
        inst✝ : DecidableEq μ
        a : ι
        s : Finset ι
        h : Not (Membership.mem s a)
        f : Finsupp ι μ
        hsupp : HasSubset.Subset f.support (Insert.insert a s)
        ⊢ Eq (s.sum ⇑(Finsupp.erase a f)) (s.sum fun x => f x)
      -/
    · apply sum_congr rfl
      /-
        case mp.intro.refine_2
        ι : Type u_1
        μ : Type u_2
        inst✝³ : DecidableEq ι
        inst✝² : AddCommMonoid μ
        inst✝¹ : Finset.HasAntidiagonal μ
        inst✝ : DecidableEq μ
        a : ι
        s : Finset ι
        h : Not (Membership.mem s a)
        f : Finsupp ι μ
        hsupp : HasSubset.Subset f.support (Insert.insert a s)
        ⊢ ∀ (x : ι), Membership.mem s x → Eq ((Finsupp.erase a f) x) (f x)
      -/
      intro x hx
      /-
        case mp.intro.refine_2
        ι : Type u_1
        μ : Type u_2
        inst✝³ : DecidableEq ι
        inst✝² : AddCommMonoid μ
        inst✝¹ : Finset.HasAntidiagonal μ
        inst✝ : DecidableEq μ
        a : ι
        s : Finset ι
        h : Not (Membership.mem s a)
        f : Finsupp ι μ
        hsupp : HasSubset.Subset f.support (Insert.insert a s)
        x : ι
        hx : Membership.mem s x
        ⊢ Eq ((Finsupp.erase a f) x) (f x)
      -/
      rw [Finsupp.erase_ne (ne_of_mem_of_not_mem hx h)]
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.refine_3
        ι : Type u_1
        μ : Type u_2
        inst✝³ : DecidableEq ι
        inst✝² : AddCommMonoid μ
        inst✝¹ : Finset.HasAntidiagonal μ
        inst✝ : DecidableEq μ
        a : ι
        s : Finset ι
        h : Not (Membership.mem s a)
        f : Finsupp ι μ
        hsupp : HasSubset.Subset f.support (Insert.insert a s)
        ⊢ HasSubset.Subset (Finsupp.erase a f).support s
      -/
    · rwa [support_erase, ← subset_insert_iff]
      /-
        🎉 no goals
      -/
    /-
      case mpr
      ι : Type u_1
      μ : Type u_2
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      a : ι
      s : Finset ι
      h : Not (Membership.mem s a)
      n : μ
      f : Finsupp ι μ
      ⊢ (Exists fun a_1 => Exists fun b => And (Eq (HAdd.hAdd a_1 b) n) (Exists fun  …
    -/
  · rintro ⟨n1, n2, rfl, g, rfl, rfl, hgsupp⟩
    /-
      case mpr.intro.intro.intro.intro.intro.intro
      ι : Type u_1
      μ : Type u_2
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      a : ι
      s : Finset ι
      h : Not (Membership.mem s a)
      n1 : μ
      g : Finsupp ι μ
      hgsupp : HasSubset.Subset g.support s
      ⊢ And (Eq (HAdd.hAdd ((g.update a n1) a) (s.sum fun x => (g.update a n1) x)) ( …
    -/
    refine ⟨?_, (support_update_subset _ _).trans (insert_subset_insert a hgsupp)⟩
    /-
      case mpr.intro.intro.intro.intro.intro.intro
      ι : Type u_1
      μ : Type u_2
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      a : ι
      s : Finset ι
      h : Not (Membership.mem s a)
      n1 : μ
      g : Finsupp ι μ
      hgsupp : HasSubset.Subset g.support s
      ⊢ Eq (HAdd.hAdd ((g.update a n1) a) (s.sum fun x => (g.update a n1) x)) (HAdd. …
    -/
    simp only [coe_update]
    /-
      case mpr.intro.intro.intro.intro.intro.intro
      ι : Type u_1
      μ : Type u_2
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      a : ι
      s : Finset ι
      h : Not (Membership.mem s a)
      n1 : μ
      g : Finsupp ι μ
      hgsupp : HasSubset.Subset g.support s
      ⊢ Eq (HAdd.hAdd (Function.update (⇑g) a n1 a) (s.sum fun x => Function.update  …
    -/
    apply congr_arg₂
      /-
        case mpr.intro.intro.intro.intro.intro.intro.hx
        ι : Type u_1
        μ : Type u_2
        inst✝³ : DecidableEq ι
        inst✝² : AddCommMonoid μ
        inst✝¹ : Finset.HasAntidiagonal μ
        inst✝ : DecidableEq μ
        a : ι
        s : Finset ι
        h : Not (Membership.mem s a)
        n1 : μ
        g : Finsupp ι μ
        hgsupp : HasSubset.Subset g.support s
        ⊢ Eq (Function.update (⇑g) a n1 a) n1
      -/
    · rw [Function.update_self]
      /-
        🎉 no goals
      -/
      /-
        case mpr.intro.intro.intro.intro.intro.intro.hy
        ι : Type u_1
        μ : Type u_2
        inst✝³ : DecidableEq ι
        inst✝² : AddCommMonoid μ
        inst✝¹ : Finset.HasAntidiagonal μ
        inst✝ : DecidableEq μ
        a : ι
        s : Finset ι
        h : Not (Membership.mem s a)
        n1 : μ
        g : Finsupp ι μ
        hgsupp : HasSubset.Subset g.support s
        ⊢ Eq (s.sum fun x => Function.update (⇑g) a n1 x) (s.sum ⇑g)
      -/
    · apply sum_congr rfl
      /-
        case mpr.intro.intro.intro.intro.intro.intro.hy
        ι : Type u_1
        μ : Type u_2
        inst✝³ : DecidableEq ι
        inst✝² : AddCommMonoid μ
        inst✝¹ : Finset.HasAntidiagonal μ
        inst✝ : DecidableEq μ
        a : ι
        s : Finset ι
        h : Not (Membership.mem s a)
        n1 : μ
        g : Finsupp ι μ
        hgsupp : HasSubset.Subset g.support s
        ⊢ ∀ (x : ι), Membership.mem s x → Eq (Function.update (⇑g) a n1 x) (g x)
      -/
      intro x hx
      /-
        case mpr.intro.intro.intro.intro.intro.intro.hy
        ι : Type u_1
        μ : Type u_2
        inst✝³ : DecidableEq ι
        inst✝² : AddCommMonoid μ
        inst✝¹ : Finset.HasAntidiagonal μ
        inst✝ : DecidableEq μ
        a : ι
        s : Finset ι
        h : Not (Membership.mem s a)
        n1 : μ
        g : Finsupp ι μ
        hgsupp : HasSubset.Subset g.support s
        x : ι
        hx : Membership.mem s x
        ⊢ Eq (Function.update (⇑g) a n1 x) (g x)
      -/
      rw [update_of_ne (ne_of_mem_of_not_mem hx h) n1 ⇑g]
      /-
        🎉 no goals
      -/


theorem finsuppAntidiag_insert {a : ι} {s : Finset ι}
    (h : a ∉ s) (n : μ) :
    finsuppAntidiag (insert a s) n = (antidiagonal n).biUnion
      (fun p : μ × μ =>
        (finsuppAntidiag s p.snd).attach.map
        ⟨fun f => Finsupp.update f.val a p.fst,
        (fun ⟨f, hf⟩ ⟨g, hg⟩ hfg => Subtype.ext <| by
          /-
            ι : Type u_1
            μ : Type u_2
            μ' : Type u_3
            inst✝³ : DecidableEq ι
            inst✝² : AddCommMonoid μ
            inst✝¹ : Finset.HasAntidiagonal μ
            inst✝ : DecidableEq μ
            s✝ : Finset ι
            n✝ : μ
            f✝ : Finsupp ι μ
            a : ι
            s : Finset ι
            h : Not (Membership.mem s a)
            n : μ
            p : Prod μ μ
            x✝¹ x✝ : Subtype fun x => Membership.mem (s.finsuppAntidiag p.2) x
            f : Finsupp ι μ
            hf : Membership.mem (s.finsuppAntidiag p.2) f
            g : Finsupp ι μ
            hg : Membership.mem (s.finsuppAntidiag p.2) g
            hfg : Eq ((fun f => (↑f).update a p.1) ⟨f, hf⟩) ((fun f => (↑f).update a p.1)  …
            ⊢ Eq ↑⟨f, hf⟩ ↑⟨g, hg⟩
          -/
          simp only [mem_val, mem_finsuppAntidiag] at hf hg
          /-
            ι : Type u_1
            μ : Type u_2
            μ' : Type u_3
            inst✝³ : DecidableEq ι
            inst✝² : AddCommMonoid μ
            inst✝¹ : Finset.HasAntidiagonal μ
            inst✝ : DecidableEq μ
            s✝ : Finset ι
            n✝ : μ
            f✝ : Finsupp ι μ
            a : ι
            s : Finset ι
            h : Not (Membership.mem s a)
            n : μ
            p : Prod μ μ
            x✝¹ x✝ : Subtype fun x => Membership.mem (s.finsuppAntidiag p.2) x
            f : Finsupp ι μ
            hf✝ : Membership.mem (s.finsuppAntidiag p.2) f
            g : Finsupp ι μ
            hg✝ : Membership.mem (s.finsuppAntidiag p.2) g
            hfg : Eq ((fun f => (↑f).update a p.1) ⟨f, hf✝⟩) ((fun f => (↑f).update a p.1) …
            hf : And (Eq (s.sum ⇑f) p.2) (HasSubset.Subset f.support s)
            hg : And (Eq (s.sum ⇑g) p.2) (HasSubset.Subset g.support s)
            ⊢ Eq ↑⟨f, hf✝⟩ ↑⟨g, hg✝⟩
          -/
          simp only [DFunLike.ext_iff] at hfg ⊢
          /-
            ι : Type u_1
            μ : Type u_2
            μ' : Type u_3
            inst✝³ : DecidableEq ι
            inst✝² : AddCommMonoid μ
            inst✝¹ : Finset.HasAntidiagonal μ
            inst✝ : DecidableEq μ
            s✝ : Finset ι
            n✝ : μ
            f✝ : Finsupp ι μ
            a : ι
            s : Finset ι
            h : Not (Membership.mem s a)
            n : μ
            p : Prod μ μ
            x✝¹ x✝ : Subtype fun x => Membership.mem (s.finsuppAntidiag p.2) x
            f : Finsupp ι μ
            hf✝ : Membership.mem (s.finsuppAntidiag p.2) f
            g : Finsupp ι μ
            hg✝ : Membership.mem (s.finsuppAntidiag p.2) g
            hf : And (Eq (s.sum ⇑f) p.2) (HasSubset.Subset f.support s)
            hg : And (Eq (s.sum ⇑g) p.2) (HasSubset.Subset g.support s)
            hfg : ∀ (x : ι), Eq ((f.update a p.1) x) ((g.update a p.1) x)
            ⊢ ∀ (x : ι), Eq (f x) (g x)
          -/
          intro x
          /-
            ι : Type u_1
            μ : Type u_2
            μ' : Type u_3
            inst✝³ : DecidableEq ι
            inst✝² : AddCommMonoid μ
            inst✝¹ : Finset.HasAntidiagonal μ
            inst✝ : DecidableEq μ
            s✝ : Finset ι
            n✝ : μ
            f✝ : Finsupp ι μ
            a : ι
            s : Finset ι
            h : Not (Membership.mem s a)
            n : μ
            p : Prod μ μ
            x✝¹ x✝ : Subtype fun x => Membership.mem (s.finsuppAntidiag p.2) x
            f : Finsupp ι μ
            hf✝ : Membership.mem (s.finsuppAntidiag p.2) f
            g : Finsupp ι μ
            hg✝ : Membership.mem (s.finsuppAntidiag p.2) g
            hf : And (Eq (s.sum ⇑f) p.2) (HasSubset.Subset f.support s)
            hg : And (Eq (s.sum ⇑g) p.2) (HasSubset.Subset g.support s)
            hfg : ∀ (x : ι), Eq ((f.update a p.1) x) ((g.update a p.1) x)
            x : ι
            ⊢ Eq (f x) (g x)
          -/
          obtain rfl | hx := eq_or_ne x a
            /-
              case inl
              ι : Type u_1
              μ : Type u_2
              μ' : Type u_3
              inst✝³ : DecidableEq ι
              inst✝² : AddCommMonoid μ
              inst✝¹ : Finset.HasAntidiagonal μ
              inst✝ : DecidableEq μ
              s✝ : Finset ι
              n✝ : μ
              f✝ : Finsupp ι μ
              s : Finset ι
              n : μ
              p : Prod μ μ
              x✝¹ x✝ : Subtype fun x => Membership.mem (s.finsuppAntidiag p.2) x
              f : Finsupp ι μ
              hf✝ : Membership.mem (s.finsuppAntidiag p.2) f
              g : Finsupp ι μ
              hg✝ : Membership.mem (s.finsuppAntidiag p.2) g
              hf : And (Eq (s.sum ⇑f) p.2) (HasSubset.Subset f.support s)
              hg : And (Eq (s.sum ⇑g) p.2) (HasSubset.Subset g.support s)
              x : ι
              h : Not (Membership.mem s x)
              hfg : ∀ (x_1 : ι), Eq ((f.update x p.1) x_1) ((g.update x p.1) x_1)
              ⊢ Eq (f x) (g x)
            -/
          · replace hf := mt (hf.2 ·) h
            /-
              case inl
              ι : Type u_1
              μ : Type u_2
              μ' : Type u_3
              inst✝³ : DecidableEq ι
              inst✝² : AddCommMonoid μ
              inst✝¹ : Finset.HasAntidiagonal μ
              inst✝ : DecidableEq μ
              s✝ : Finset ι
              n✝ : μ
              f✝ : Finsupp ι μ
              s : Finset ι
              n : μ
              p : Prod μ μ
              x✝¹ x✝ : Subtype fun x => Membership.mem (s.finsuppAntidiag p.2) x
              f : Finsupp ι μ
              hf✝ : Membership.mem (s.finsuppAntidiag p.2) f
              g : Finsupp ι μ
              hg✝ : Membership.mem (s.finsuppAntidiag p.2) g
              hg : And (Eq (s.sum ⇑g) p.2) (HasSubset.Subset g.support s)
              x : ι
              h : Not (Membership.mem s x)
              hfg : ∀ (x_1 : ι), Eq ((f.update x p.1) x_1) ((g.update x p.1) x_1)
              hf : Not (Membership.mem f.support x)
              ⊢ Eq (f x) (g x)
            -/
            replace hg := mt (hg.2 ·) h
            /-
              case inl
              ι : Type u_1
              μ : Type u_2
              μ' : Type u_3
              inst✝³ : DecidableEq ι
              inst✝² : AddCommMonoid μ
              inst✝¹ : Finset.HasAntidiagonal μ
              inst✝ : DecidableEq μ
              s✝ : Finset ι
              n✝ : μ
              f✝ : Finsupp ι μ
              s : Finset ι
              n : μ
              p : Prod μ μ
              x✝¹ x✝ : Subtype fun x => Membership.mem (s.finsuppAntidiag p.2) x
              f : Finsupp ι μ
              hf✝ : Membership.mem (s.finsuppAntidiag p.2) f
              g : Finsupp ι μ
              hg✝ : Membership.mem (s.finsuppAntidiag p.2) g
              x : ι
              h : Not (Membership.mem s x)
              hfg : ∀ (x_1 : ι), Eq ((f.update x p.1) x_1) ((g.update x p.1) x_1)
              hf : Not (Membership.mem f.support x)
              hg : Not (Membership.mem g.support x)
              ⊢ Eq (f x) (g x)
            -/
            rw [not_mem_support_iff.mp hf, not_mem_support_iff.mp hg]
            /-
              🎉 no goals
            -/
            /-
              case inr
              ι : Type u_1
              μ : Type u_2
              μ' : Type u_3
              inst✝³ : DecidableEq ι
              inst✝² : AddCommMonoid μ
              inst✝¹ : Finset.HasAntidiagonal μ
              inst✝ : DecidableEq μ
              s✝ : Finset ι
              n✝ : μ
              f✝ : Finsupp ι μ
              a : ι
              s : Finset ι
              h : Not (Membership.mem s a)
              n : μ
              p : Prod μ μ
              x✝¹ x✝ : Subtype fun x => Membership.mem (s.finsuppAntidiag p.2) x
              f : Finsupp ι μ
              hf✝ : Membership.mem (s.finsuppAntidiag p.2) f
              g : Finsupp ι μ
              hg✝ : Membership.mem (s.finsuppAntidiag p.2) g
              hf : And (Eq (s.sum ⇑f) p.2) (HasSubset.Subset f.support s)
              hg : And (Eq (s.sum ⇑g) p.2) (HasSubset.Subset g.support s)
              hfg : ∀ (x : ι), Eq ((f.update a p.1) x) ((g.update a p.1) x)
              x : ι
              hx : Ne x a
              ⊢ Eq (f x) (g x)
            -/
          · simpa only [coe_update, Function.update, dif_neg hx] using hfg x)⟩) := by
            /-
              🎉 no goals
            -/
  /-
    ι : Type u_1
    μ : Type u_2
    inst✝³ : DecidableEq ι
    inst✝² : AddCommMonoid μ
    inst✝¹ : Finset.HasAntidiagonal μ
    inst✝ : DecidableEq μ
    a : ι
    s : Finset ι
    h : Not (Membership.mem s a)
    n : μ
    ⊢ Eq ((Insert.insert a s).finsuppAntidiag n) ((Finset.HasAntidiagonal.antidiag …
  -/
  ext f
  /-
    case h
    ι : Type u_1
    μ : Type u_2
    inst✝³ : DecidableEq ι
    inst✝² : AddCommMonoid μ
    inst✝¹ : Finset.HasAntidiagonal μ
    inst✝ : DecidableEq μ
    a : ι
    s : Finset ι
    h : Not (Membership.mem s a)
    n : μ
    f : Finsupp ι μ
    ⊢ Iff (Membership.mem ((Insert.insert a s).finsuppAntidiag n) f) (Membership.m …
  -/
  rw [mem_finsuppAntidiag_insert h, mem_biUnion]
  simp_rw [mem_map, mem_attach, true_and, Subtype.exists, Embedding.coeFn_mk, exists_prop, and_comm,
    eq_comm]


lemma mapRange_finsuppAntidiag_subset {e : μ ≃+ μ'} {s : Finset ι} {n : μ} :
    (finsuppAntidiag s n).map (mapRange.addEquiv e).toEmbedding ⊆ finsuppAntidiag s (e n) := by
  /-
    ι : Type u_1
    μ : Type u_2
    μ' : Type u_3
    inst✝⁶ : DecidableEq ι
    inst✝⁵ : AddCommMonoid μ
    inst✝⁴ : Finset.HasAntidiagonal μ
    inst✝³ : DecidableEq μ
    inst✝² : AddCommMonoid μ'
    inst✝¹ : Finset.HasAntidiagonal μ'
    inst✝ : DecidableEq μ'
    e : AddEquiv μ μ'
    s : Finset ι
    n : μ
    ⊢ HasSubset.Subset (Finset.map (Finsupp.mapRange.addEquiv e).toEmbedding (s.fi …
  -/
  intro f
  /-
    ι : Type u_1
    μ : Type u_2
    μ' : Type u_3
    inst✝⁶ : DecidableEq ι
    inst✝⁵ : AddCommMonoid μ
    inst✝⁴ : Finset.HasAntidiagonal μ
    inst✝³ : DecidableEq μ
    inst✝² : AddCommMonoid μ'
    inst✝¹ : Finset.HasAntidiagonal μ'
    inst✝ : DecidableEq μ'
    e : AddEquiv μ μ'
    s : Finset ι
    n : μ
    f : Finsupp ι μ'
    ⊢ Membership.mem (Finset.map (Finsupp.mapRange.addEquiv e).toEmbedding (s.fins …
  -/
  simp only [mem_map, mem_finsuppAntidiag']
  /-
    ι : Type u_1
    μ : Type u_2
    μ' : Type u_3
    inst✝⁶ : DecidableEq ι
    inst✝⁵ : AddCommMonoid μ
    inst✝⁴ : Finset.HasAntidiagonal μ
    inst✝³ : DecidableEq μ
    inst✝² : AddCommMonoid μ'
    inst✝¹ : Finset.HasAntidiagonal μ'
    inst✝ : DecidableEq μ'
    e : AddEquiv μ μ'
    s : Finset ι
    n : μ
    f : Finsupp ι μ'
    ⊢ (Exists fun a => And (And (Eq (a.sum fun x x => x) n) (HasSubset.Subset a.su …
  -/
  rintro ⟨g, ⟨hsum, hsupp⟩, rfl⟩
  simp only [AddEquiv.toEquiv_eq_coe, mapRange.addEquiv_toEquiv, Equiv.coe_toEmbedding,
    mapRange.equiv_apply, EquivLike.coe_coe]
  /-
    case intro.intro.intro
    ι : Type u_1
    μ : Type u_2
    μ' : Type u_3
    inst✝⁶ : DecidableEq ι
    inst✝⁵ : AddCommMonoid μ
    inst✝⁴ : Finset.HasAntidiagonal μ
    inst✝³ : DecidableEq μ
    inst✝² : AddCommMonoid μ'
    inst✝¹ : Finset.HasAntidiagonal μ'
    inst✝ : DecidableEq μ'
    e : AddEquiv μ μ'
    s : Finset ι
    n : μ
    g : Finsupp ι μ
    hsum : Eq (g.sum fun x x => x) n
    hsupp : HasSubset.Subset g.support s
    ⊢ And (Eq ((Finsupp.mapRange ⇑e ⋯ g).sum fun x x => x) (e n)) (HasSubset.Subse …
  -/
  constructor
    /-
      case intro.intro.intro.left
      ι : Type u_1
      μ : Type u_2
      μ' : Type u_3
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : AddCommMonoid μ
      inst✝⁴ : Finset.HasAntidiagonal μ
      inst✝³ : DecidableEq μ
      inst✝² : AddCommMonoid μ'
      inst✝¹ : Finset.HasAntidiagonal μ'
      inst✝ : DecidableEq μ'
      e : AddEquiv μ μ'
      s : Finset ι
      n : μ
      g : Finsupp ι μ
      hsum : Eq (g.sum fun x x => x) n
      hsupp : HasSubset.Subset g.support s
      ⊢ Eq ((Finsupp.mapRange ⇑e ⋯ g).sum fun x x => x) (e n)
    -/
  · rw [sum_mapRange_index (fun _ ↦ rfl), ← hsum, _root_.map_finsupp_sum]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.right
      ι : Type u_1
      μ : Type u_2
      μ' : Type u_3
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : AddCommMonoid μ
      inst✝⁴ : Finset.HasAntidiagonal μ
      inst✝³ : DecidableEq μ
      inst✝² : AddCommMonoid μ'
      inst✝¹ : Finset.HasAntidiagonal μ'
      inst✝ : DecidableEq μ'
      e : AddEquiv μ μ'
      s : Finset ι
      n : μ
      g : Finsupp ι μ
      hsum : Eq (g.sum fun x x => x) n
      hsupp : HasSubset.Subset g.support s
      ⊢ HasSubset.Subset (Finsupp.mapRange ⇑e ⋯ g).support s
    -/
  · exact subset_trans (support_mapRange) hsupp
    /-
      🎉 no goals
    -/


lemma mapRange_finsuppAntidiag_eq {e : μ ≃+ μ'} {s : Finset ι} {n : μ} :
    (finsuppAntidiag s n).map (mapRange.addEquiv e).toEmbedding = finsuppAntidiag s (e n) := by
  /-
    ι : Type u_1
    μ : Type u_2
    μ' : Type u_3
    inst✝⁶ : DecidableEq ι
    inst✝⁵ : AddCommMonoid μ
    inst✝⁴ : Finset.HasAntidiagonal μ
    inst✝³ : DecidableEq μ
    inst✝² : AddCommMonoid μ'
    inst✝¹ : Finset.HasAntidiagonal μ'
    inst✝ : DecidableEq μ'
    e : AddEquiv μ μ'
    s : Finset ι
    n : μ
    ⊢ Eq (Finset.map (Finsupp.mapRange.addEquiv e).toEmbedding (s.finsuppAntidiag  …
  -/
  ext f
  /-
    case h
    ι : Type u_1
    μ : Type u_2
    μ' : Type u_3
    inst✝⁶ : DecidableEq ι
    inst✝⁵ : AddCommMonoid μ
    inst✝⁴ : Finset.HasAntidiagonal μ
    inst✝³ : DecidableEq μ
    inst✝² : AddCommMonoid μ'
    inst✝¹ : Finset.HasAntidiagonal μ'
    inst✝ : DecidableEq μ'
    e : AddEquiv μ μ'
    s : Finset ι
    n : μ
    f : Finsupp ι μ'
    ⊢ Iff (Membership.mem (Finset.map (Finsupp.mapRange.addEquiv e).toEmbedding (s …
  -/
  constructor
    /-
      case h.mp
      ι : Type u_1
      μ : Type u_2
      μ' : Type u_3
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : AddCommMonoid μ
      inst✝⁴ : Finset.HasAntidiagonal μ
      inst✝³ : DecidableEq μ
      inst✝² : AddCommMonoid μ'
      inst✝¹ : Finset.HasAntidiagonal μ'
      inst✝ : DecidableEq μ'
      e : AddEquiv μ μ'
      s : Finset ι
      n : μ
      f : Finsupp ι μ'
      ⊢ Membership.mem (Finset.map (Finsupp.mapRange.addEquiv e).toEmbedding (s.fins …
    -/
  · apply mapRange_finsuppAntidiag_subset
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      ι : Type u_1
      μ : Type u_2
      μ' : Type u_3
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : AddCommMonoid μ
      inst✝⁴ : Finset.HasAntidiagonal μ
      inst✝³ : DecidableEq μ
      inst✝² : AddCommMonoid μ'
      inst✝¹ : Finset.HasAntidiagonal μ'
      inst✝ : DecidableEq μ'
      e : AddEquiv μ μ'
      s : Finset ι
      n : μ
      f : Finsupp ι μ'
      ⊢ Membership.mem (s.finsuppAntidiag (e n)) f → Membership.mem (Finset.map (Fin …
    -/
  · set h := (mapRange.addEquiv e).toEquiv with hh
    /-
      case h.mpr
      ι : Type u_1
      μ : Type u_2
      μ' : Type u_3
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : AddCommMonoid μ
      inst✝⁴ : Finset.HasAntidiagonal μ
      inst✝³ : DecidableEq μ
      inst✝² : AddCommMonoid μ'
      inst✝¹ : Finset.HasAntidiagonal μ'
      inst✝ : DecidableEq μ'
      e : AddEquiv μ μ'
      s : Finset ι
      n : μ
      f : Finsupp ι μ'
      h : Equiv (Finsupp ι μ) (Finsupp ι μ') := (Finsupp.mapRange.addEquiv e).toEquiv
      hh : Eq h (Finsupp.mapRange.addEquiv e).toEquiv
      ⊢ Membership.mem (s.finsuppAntidiag (e n)) f → Membership.mem (Finset.map h.to …
    -/
    intro hf
    /-
      case h.mpr
      ι : Type u_1
      μ : Type u_2
      μ' : Type u_3
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : AddCommMonoid μ
      inst✝⁴ : Finset.HasAntidiagonal μ
      inst✝³ : DecidableEq μ
      inst✝² : AddCommMonoid μ'
      inst✝¹ : Finset.HasAntidiagonal μ'
      inst✝ : DecidableEq μ'
      e : AddEquiv μ μ'
      s : Finset ι
      n : μ
      f : Finsupp ι μ'
      h : Equiv (Finsupp ι μ) (Finsupp ι μ') := (Finsupp.mapRange.addEquiv e).toEquiv
      hh : Eq h (Finsupp.mapRange.addEquiv e).toEquiv
      hf : Membership.mem (s.finsuppAntidiag (e n)) f
      ⊢ Membership.mem (Finset.map h.toEmbedding (s.finsuppAntidiag n)) f
    -/
    have : n = e.symm (e n) := (AddEquiv.eq_symm_apply e).mpr rfl
    /-
      case h.mpr
      ι : Type u_1
      μ : Type u_2
      μ' : Type u_3
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : AddCommMonoid μ
      inst✝⁴ : Finset.HasAntidiagonal μ
      inst✝³ : DecidableEq μ
      inst✝² : AddCommMonoid μ'
      inst✝¹ : Finset.HasAntidiagonal μ'
      inst✝ : DecidableEq μ'
      e : AddEquiv μ μ'
      s : Finset ι
      n : μ
      f : Finsupp ι μ'
      h : Equiv (Finsupp ι μ) (Finsupp ι μ') := (Finsupp.mapRange.addEquiv e).toEquiv
      hh : Eq h (Finsupp.mapRange.addEquiv e).toEquiv
      hf : Membership.mem (s.finsuppAntidiag (e n)) f
      this : Eq n (e.symm (e n))
      ⊢ Membership.mem (Finset.map h.toEmbedding (s.finsuppAntidiag n)) f
    -/
    rw [mem_map_equiv, this]
    /-
      case h.mpr
      ι : Type u_1
      μ : Type u_2
      μ' : Type u_3
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : AddCommMonoid μ
      inst✝⁴ : Finset.HasAntidiagonal μ
      inst✝³ : DecidableEq μ
      inst✝² : AddCommMonoid μ'
      inst✝¹ : Finset.HasAntidiagonal μ'
      inst✝ : DecidableEq μ'
      e : AddEquiv μ μ'
      s : Finset ι
      n : μ
      f : Finsupp ι μ'
      h : Equiv (Finsupp ι μ) (Finsupp ι μ') := (Finsupp.mapRange.addEquiv e).toEquiv
      hh : Eq h (Finsupp.mapRange.addEquiv e).toEquiv
      hf : Membership.mem (s.finsuppAntidiag (e n)) f
      this : Eq n (e.symm (e n))
      ⊢ Membership.mem (s.finsuppAntidiag (e.symm (e n))) (h.symm f)
    -/
    apply mapRange_finsuppAntidiag_subset
    /-
      case h.mpr.a
      ι : Type u_1
      μ : Type u_2
      μ' : Type u_3
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : AddCommMonoid μ
      inst✝⁴ : Finset.HasAntidiagonal μ
      inst✝³ : DecidableEq μ
      inst✝² : AddCommMonoid μ'
      inst✝¹ : Finset.HasAntidiagonal μ'
      inst✝ : DecidableEq μ'
      e : AddEquiv μ μ'
      s : Finset ι
      n : μ
      f : Finsupp ι μ'
      h : Equiv (Finsupp ι μ) (Finsupp ι μ') := (Finsupp.mapRange.addEquiv e).toEquiv
      hh : Eq h (Finsupp.mapRange.addEquiv e).toEquiv
      hf : Membership.mem (s.finsuppAntidiag (e n)) f
      this : Eq n (e.symm (e n))
      ⊢ Membership.mem (Finset.map (Finsupp.mapRange.addEquiv e.symm).toEmbedding (s …
    -/
    rw [← mem_map_equiv]
    /-
      case h.mpr.a
      ι : Type u_1
      μ : Type u_2
      μ' : Type u_3
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : AddCommMonoid μ
      inst✝⁴ : Finset.HasAntidiagonal μ
      inst✝³ : DecidableEq μ
      inst✝² : AddCommMonoid μ'
      inst✝¹ : Finset.HasAntidiagonal μ'
      inst✝ : DecidableEq μ'
      e : AddEquiv μ μ'
      s : Finset ι
      n : μ
      f : Finsupp ι μ'
      h : Equiv (Finsupp ι μ) (Finsupp ι μ') := (Finsupp.mapRange.addEquiv e).toEquiv
      hh : Eq h (Finsupp.mapRange.addEquiv e).toEquiv
      hf : Membership.mem (s.finsuppAntidiag (e n)) f
      this : Eq n (e.symm (e n))
      ⊢ Membership.mem (Finset.map h.toEmbedding (Finset.map (Finsupp.mapRange.addEq …
    -/
    convert hf
    /-
      case h.e'_4
      ι : Type u_1
      μ : Type u_2
      μ' : Type u_3
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : AddCommMonoid μ
      inst✝⁴ : Finset.HasAntidiagonal μ
      inst✝³ : DecidableEq μ
      inst✝² : AddCommMonoid μ'
      inst✝¹ : Finset.HasAntidiagonal μ'
      inst✝ : DecidableEq μ'
      e : AddEquiv μ μ'
      s : Finset ι
      n : μ
      f : Finsupp ι μ'
      h : Equiv (Finsupp ι μ) (Finsupp ι μ') := (Finsupp.mapRange.addEquiv e).toEquiv
      hh : Eq h (Finsupp.mapRange.addEquiv e).toEquiv
      hf : Membership.mem (s.finsuppAntidiag (e n)) f
      this : Eq n (e.symm (e n))
      ⊢ Eq (Finset.map h.toEmbedding (Finset.map (Finsupp.mapRange.addEquiv e.symm). …
    -/
    rw [map_map, hh]
    /-
      case h.e'_4
      ι : Type u_1
      μ : Type u_2
      μ' : Type u_3
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : AddCommMonoid μ
      inst✝⁴ : Finset.HasAntidiagonal μ
      inst✝³ : DecidableEq μ
      inst✝² : AddCommMonoid μ'
      inst✝¹ : Finset.HasAntidiagonal μ'
      inst✝ : DecidableEq μ'
      e : AddEquiv μ μ'
      s : Finset ι
      n : μ
      f : Finsupp ι μ'
      h : Equiv (Finsupp ι μ) (Finsupp ι μ') := (Finsupp.mapRange.addEquiv e).toEquiv
      hh : Eq h (Finsupp.mapRange.addEquiv e).toEquiv
      hf : Membership.mem (s.finsuppAntidiag (e n)) f
      this : Eq n (e.symm (e n))
      ⊢ Eq (Finset.map ((Finsupp.mapRange.addEquiv e.symm).toEmbedding.trans (Finsup …
    -/
    convert map_refl
    /-
      case h.e'_2.h.e'_3
      ι : Type u_1
      μ : Type u_2
      μ' : Type u_3
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : AddCommMonoid μ
      inst✝⁴ : Finset.HasAntidiagonal μ
      inst✝³ : DecidableEq μ
      inst✝² : AddCommMonoid μ'
      inst✝¹ : Finset.HasAntidiagonal μ'
      inst✝ : DecidableEq μ'
      e : AddEquiv μ μ'
      s : Finset ι
      n : μ
      f : Finsupp ι μ'
      h : Equiv (Finsupp ι μ) (Finsupp ι μ') := (Finsupp.mapRange.addEquiv e).toEquiv
      hh : Eq h (Finsupp.mapRange.addEquiv e).toEquiv
      hf : Membership.mem (s.finsuppAntidiag (e n)) f
      this : Eq n (e.symm (e n))
      ⊢ Eq ((Finsupp.mapRange.addEquiv e.symm).toEmbedding.trans (Finsupp.mapRange.a …
    -/
    apply Function.Embedding.equiv_symm_toEmbedding_trans_toEmbedding
    /-
      🎉 no goals
    -/


@[simp] lemma finsuppAntidiag_zero (s : Finset ι) : finsuppAntidiag s (0 : μ) = {0} := by
  /-
    ι : Type u_1
    μ : Type u_2
    inst✝³ : DecidableEq ι
    inst✝² : DecidableEq μ
    inst✝¹ : CanonicallyOrderedAddCommMonoid μ
    inst✝ : Finset.HasAntidiagonal μ
    s : Finset ι
    ⊢ Eq (s.finsuppAntidiag 0) (Singleton.singleton 0)
  -/
  ext f; simp [finsuppAntidiag, ← DFunLike.coe_fn_eq (g := f), -mem_piAntidiag, eq_comm]
         /-
           🎉 no goals
         -/


