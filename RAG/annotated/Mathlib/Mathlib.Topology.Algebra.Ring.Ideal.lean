/-- The closure of an ideal in a topological ring as an ideal. -/
protected def Ideal.closure (I : Ideal R) : Ideal R :=
  {
    AddSubmonoid.topologicalClosure
      I.toAddSubmonoid with
    carrier := closure I
    smul_mem' := fun c _ hx => map_mem_closure (mulLeft_continuous _) hx fun _ => I.mul_mem_left c }


@[simp]
theorem Ideal.coe_closure (I : Ideal R) : (I.closure : Set R) = closure I :=
  rfl

-- Porting note: removed `@[simp]` because we make the instance argument explicit since otherwise
-- it causes timeouts as `simp` tries and fails to generated an `IsClosed` instance.
-- https://leanprover.zulipchat.com/#narrow/stream/287929-mathlib4/topic/!4.234852.20heartbeats.20of.20the.20linter

theorem Ideal.closure_eq_of_isClosed (I : Ideal R) (hI : IsClosed (I : Set R)) : I.closure = I :=
  SetLike.ext' hI.closure_eq


instance topologicalRingQuotientTopology : TopologicalSpace (R ⧸ N) :=
  instTopologicalSpaceQuotient

-- note for the reader: in the following, `mk` is `Ideal.Quotient.mk`, the canonical map `R → R/I`.

theorem QuotientRing.isOpenMap_coe : IsOpenMap (mk N) :=
  QuotientAddGroup.isOpenMap_coe


theorem QuotientRing.isOpenQuotientMap_mk : IsOpenQuotientMap (mk N) :=
  QuotientAddGroup.isOpenQuotientMap_mk


theorem QuotientRing.isQuotientMap_coe_coe : IsQuotientMap fun p : R × R => (mk N p.1, mk N p.2) :=
  ((isOpenQuotientMap_mk N).prodMap (isOpenQuotientMap_mk N)).isQuotientMap


@[deprecated (since := "2024-10-22")]
alias QuotientRing.quotientMap_coe_coe := QuotientRing.isQuotientMap_coe_coe


instance topologicalRing_quotient : TopologicalRing (R ⧸ N) where
  __ := QuotientAddGroup.instTopologicalAddGroup _
  continuous_mul := (QuotientRing.isQuotientMap_coe_coe N).continuous_iff.2 <|
    continuous_quot_mk.comp continuous_mul


